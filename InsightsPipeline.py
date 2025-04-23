#%%
import pandas as pd
from scipy.stats import skew, kurtosis
from pandas.api.types import is_object_dtype, is_categorical_dtype,is_numeric_dtype
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.cluster import MiniBatchKMeans
#%%
def infer_best_datetime(s, threshold=0.6):
    """Return kwargs for pd.to_datetime that successfully parses ≥threshold of s."""
    configs = [
        {'infer_datetime_format': True, 'dayfirst': False, 'yearfirst': False},  # Changed dict() to {}
        {'infer_datetime_format': True, 'dayfirst': True},  # Changed dict() to {}
        {'infer_datetime_format': True, 'yearfirst': True},  # Changed dict() to {}
        {'format': "%Y-%m-%d %H:%M:%S", 'utc': True},  # Changed dict() to {}
        {'format': "%Y-%m-%d"},  # Changed dict() to {} # added common date format
        {'format': "%m/%d/%Y"},  # Changed dict() to {} # added more flexible date format
    ]
    best = (None, 0.0)
    for kwargs in configs:
        try:
            parsed = pd.to_datetime(s, errors="coerce", **kwargs)
            rate = parsed.notna().mean()
            if rate > best[1]:
                best = (kwargs, rate)
        except (ValueError, TypeError):
            continue

    return best[0] if best[1] >= threshold else None


def smart_type_inference(df, threshold=0.6, logs=None):
    if logs is None:
        logs = []

    df = df.copy()

    # 0) Catch any obvious numerics in object columns up‐front
    df = df.infer_objects()

    for col in df.select_dtypes(include="object"):
        s = df[col].dropna().astype(str).str.strip().head(1000)
        if len(s) == 0:
            continue

        # 1) Pure time? (HH:MM:SS)
        if s.str.match(r'^\d{2}:\d{2}:\d{2}$').mean() >= threshold:
            df[col] = pd.to_datetime(df[col], format='%H:%M:%S',
                                     errors='coerce').dt.time
            logs.append(f"Inferred column '{col}' as time.")
            continue

        # 2) Datetime variants
        best_cfg = infer_best_datetime(s, threshold=threshold)
        if best_cfg is not None:
            df[col] = pd.to_datetime(df[col], errors="coerce", **best_cfg)
            logs.append(f"Inferred column '{col}' as datetime using format: {best_cfg}")

            # Extract Year, Month, Day, and drop the original column
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df.drop(columns=[col], inplace=True)  # Drop the original column

            continue

        # 3) Numeric?
        numeric_pattern = r'^\(?[$€£₹]?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?%?$'
        match_mask = s.str.match(numeric_pattern, na=False)
        if match_mask.mean() >= threshold:
            cleaned = df[col].astype(str).str.replace(r'[\$,€£₹()%]', '', regex=True)
            cleaned = cleaned.str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(cleaned, errors='coerce')
            logs.append(f"Inferred column '{col}' as numeric.")
            continue

        # 4) Boolean-ish?
        if s.str.lower().isin(['true', 'false', 'yes', 'no', '1', '0']).mean() >= threshold:
            df[col] = s.str.lower().map({
                'true': True, 'false': False,
                'yes': True, 'no': False,
                '1': True, '0': False
            })
            logs.append(f"Inferred column '{col}' as boolean.")
            continue

    return df
#%%
def profile_column(df: pd.DataFrame, col: str) -> dict:
    """
    Generate profile statistics for a single column.
    Returns a dictionary of dtype, missing ratio, and summary stats.
    """
    col_data = df[col].dropna()
    col_type = df[col].dtype
    missing_ratio = df[col].isnull().mean()
    entry = {
        'dtype': str(col_type),
        'missing_ratio': round(missing_ratio, 4)
    }

    if pd.api.types.is_numeric_dtype(col_type):
        entry.update({
            'mean': col_data.mean(),
            'median': col_data.median(),
            'std_dev': col_data.std(),
            'min': col_data.min(),
            'max': col_data.max(),
            'range': col_data.max() - col_data.min(),
            'variance': col_data.var(),
            'skewness': skew(col_data),
            'kurtosis': kurtosis(col_data),
        })
        # maximum absolute correlation with other numeric columns
        try:
            corr_series = df.corr(numeric_only=True)[col].drop(col).abs()
            entry['correlation_max'] = corr_series.max() if not corr_series.empty else None
        except Exception:
            entry['correlation_max'] = None

    elif pd.api.types.is_object_dtype(col_type) or pd.api.types.is_categorical_dtype(col_type):
        num_unique = df[col].nunique()
        entry['cardinality'] = num_unique
        entry['is_categorical_like'] = num_unique < 50

    return entry
#%%
def profile_dataframe(df: pd.DataFrame) -> dict:
    """
    Generate a profiling report for all columns in df.
    Returns a dictionary keyed by column names.
    """
    profile_report = {}
    for col in df.columns:
        profile_report[col] = profile_column(df, col)
    return {'columns': profile_report}

#%%
def detectOutliers(df: pd.DataFrame, contamination: float = 0.05, random_state: int = 42):
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    preds = iso_forest.fit_predict(numeric_df)

    outlier_indices = df.index[preds == -1].tolist()

    return {'outlier_indices': outlier_indices}


def detectOutlierRows(df: pd.DataFrame):
    outlier_rows = set()
    outliers = detectOutliers(df)  # Now a dict
    for indices in outliers.values():
        outlier_rows.update(indices)
    return list(outlier_rows)
#%%
def simulate_feature_importance(
    df: pd.DataFrame,
    random_state: int = 42
) -> dict:
    """
    Estimate feature importance for each column as target by training a Decision Tree on a dynamically downsampled dataset.

    Sampling strategy based on dataset size:
    - < 5,000 rows: use full dataset
    - 5,000–19,999 rows: sample 50%
    - ≥ 20,000 rows: sample 25%

    Parameters:
    - df: Original DataFrame.
    - random_state: Seed for reproducibility.

    Returns:
    - Dictionary containing predictability score and top 3 predictors for each target column.
    """
    # 1. Determine dynamic sample fraction based on size
    n_rows = len(df)
    if n_rows < 5000:
        sample_frac = 1.0
    elif n_rows < 20000:
        sample_frac = 0.5
    else:
        sample_frac = 0.25

    # 2. Downsample the DataFrame for faster computation
    df_sample = df.sample(frac=sample_frac, random_state=random_state).copy()

    # 3. Encode categorical columns
    label_encoders = {}
    df_encoded = df_sample.copy()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df_encoded[col]):
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le

    results = []
    for target_col in df_encoded.columns:
        X = df_encoded.drop(columns=[target_col])
        y = df_encoded[target_col]
        is_numeric = pd.api.types.is_numeric_dtype(y)
        model = DecisionTreeRegressor(random_state=random_state) if is_numeric else DecisionTreeClassifier(random_state=random_state)

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_state
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            score = r2_score(y_test, y_pred) if is_numeric else accuracy_score(y_test, y_pred)
            score = max(score, 0)

            top_predictors = sorted(
                zip(X.columns, model.feature_importances_),
                key=lambda x: x[1],
                reverse=True
            )[:3]

            results.append({
                "target": target_col,
                "type": "numeric" if is_numeric else "categorical",
                "predictability_score": round(score, 3),
                "top_predictors": [
                    {"feature": feat, "importance": round(impt, 3)} for feat, impt in top_predictors
                ]
            })

        except Exception as e:
            print(f"Failed for column {target_col}: {e}")
            continue

    return {"feature_importance": results}

#%%
def encode_non_numeric(df, epsilon=0.01):
    df_encoded = df.copy()
    categorical_columns = df_encoded.select_dtypes(include=[object, 'category']).columns

    # Frequency Encoding for non-numeric columns with random perturbation
    for col in categorical_columns:
        freq_encoding = df_encoded[col].value_counts() / len(df_encoded)
        df_encoded[col] = df_encoded[col].map(freq_encoding)

        # Add small random perturbation to break ties if encodings are identical
        df_encoded[col] += np.random.uniform(-epsilon, epsilon, size=len(df_encoded))

    return df_encoded

def remove_useless_features(df):
    # Remove columns with constant values
    constant_columns = [col for col in df.columns if df[col].nunique() == 1]
    df = df.drop(columns=constant_columns)

    # Remove columns with too many missing values (threshold of 50% missing)
    missing_columns = df.columns[df.isnull().mean() > 0.5]
    df = df.drop(columns=missing_columns)

    # Correlation analysis only on numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()

    correlated_features = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.9:
                colname = corr_matrix.columns[i]
                correlated_features.add(colname)

    df = df.drop(columns=correlated_features, errors='ignore')
    return df

def downsample_data(df):
    # Get the number of rows
    num_rows = len(df)

    # Apply downsampling strategy
    if num_rows <= 10000:
        return df  # No downsampling, return full dataset
    elif 10000 < num_rows <= 20000:
        return df.sample(frac=0.5, random_state=42)  # Downsample to 50%
    else:
        return df.sample(frac=0.25, random_state=42)  # Downsample to 25%

def determine_optimal_clusters(scaled_data):
    inertia = []
    silhouette_scores = []
    ks = [2, 3, 4, 5, 6]  # fewer options to speed up


    sampled_data = resample(scaled_data, n_samples=min(1000, len(scaled_data)), random_state=42)

    for k in ks:
        minibatch_kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, max_iter=100, batch_size=512)
        minibatch_kmeans.fit(scaled_data)

        inertia.append(minibatch_kmeans.inertia_)
        silhouette_avg = silhouette_score(sampled_data, minibatch_kmeans.predict(sampled_data))
        silhouette_scores.append(silhouette_avg)

    optimal_k = ks[silhouette_scores.index(max(silhouette_scores))]
    print(f"Optimal number of clusters (based on silhouette score): {optimal_k}")

    return optimal_k

def apply_mini_batch_kmeans(df):
    # 1. Feature Engineering
    # Step 1: Remove useless features (constant, too many missing, or highly correlated features)
    df_cleaned = remove_useless_features(df)

    # Step 2: Encode non-numeric columns (optimized)
    df_encoded = encode_non_numeric(df_cleaned)
    non_numeric_cols = df_encoded.select_dtypes(exclude=[np.number]).columns
    if not non_numeric_cols.empty:
        print("Non-numeric columns remaining after encoding:", non_numeric_cols.tolist())

    # Step 3: Downsample the data based on row count
    df_encoded = downsample_data(df_encoded)

    # Step 4: Select only numeric columns
    df_numeric = df_encoded.select_dtypes(include=[np.number])

    # Safety check
    if df_numeric.select_dtypes(exclude=[np.number]).shape[1] > 0:
        raise ValueError("Non-numeric columns still present before scaling!")

    # 2. Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)

    # 3. Determine the optimal number of clusters (automatically)
    optimal_k = determine_optimal_clusters(scaled_data)

    # 4. Apply MiniBatchKMeans with the optimal number of clusters
    minibatch_kmeans = MiniBatchKMeans(n_clusters=optimal_k, random_state=42)
    df_encoded['cluster'] = minibatch_kmeans.fit_predict(scaled_data)

    cluster_sizes = df_encoded['cluster'].value_counts().to_dict()

    # 5. Generate Insights from the Clusters
    cluster_centers = pd.DataFrame(minibatch_kmeans.cluster_centers_, columns=df_numeric.columns)
    print("Cluster Centers:\n", cluster_centers)

    # Descriptive statistics for each cluster
    cluster_stats = df_encoded.groupby('cluster').agg(['mean', 'std', 'min', 'max'])
    print("\nCluster Statistics:\n", cluster_stats)

    # 6. Visualize the clusters
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(scaled_data)

    df_encoded['PCA1'] = pca_components[:, 0]
    df_encoded['PCA2'] = pca_components[:, 1]

    plt.figure(figsize=(8,6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='cluster', data=df_encoded, palette='Set1', s=100)
    plt.title('Clusters Visualized with PCA')
    plt.show()

    results = {
        'method': 'MiniBatchKMeans',
        'num_clusters': optimal_k,
        'cluster_sizes': {f'Cluster {k}': v for k, v in cluster_sizes.items()},
        'features_used': ['PCA1', 'PCA2'],
        'clustered_data': df_encoded,  # Includes cluster labels and PCA coords
        'cluster_centers': cluster_centers,
        'cluster_statistics': cluster_stats
    }

    return results

#%%
def generate_column_insight(df: pd.DataFrame, columns) -> dict:
    """
    Generate profiling insights for specified columns.
    `columns` must be a list of column names, a single column name as string.
    """
    # Normalize columns parameter
    if isinstance(columns, str):
        columns = [columns]
    elif isinstance(columns, (list, tuple, set)):
        columns = list(columns)
    else:
        raise ValueError("`columns` parameter must be a list of column names, a string, or None.")

    # Validate column existence
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    # Generate insights
    insights = {}
    for col in columns:
        insights[col] = profile_column(df, col)
    return insights
#%%
def generate_full_insight(df, df_cleaned):
    """
    Generate profiling insights for both the original and cleaned DataFrames.
    """
    df = smart_type_inference(df)
    df_cleaned = smart_type_inference(df_cleaned)

    return {
        'profile': profile_dataframe(df),
        'outliers': detectOutlierRows(df),
        'feature_importance': simulate_feature_importance(df_cleaned),
        'clustering': apply_mini_batch_kmeans(df_cleaned)
    }
#%%
def insightsGenerationPipeline(df_noise, df_cleaned, columns=None):
    if columns is None:
        return generate_full_insight(df_noise, df_cleaned)
    else:
        return generate_column_insight(df_noise, columns)