#%%
import pandas as pd
from scipy.stats import skew, kurtosis
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_object_dtype, is_categorical_dtype,is_numeric_dtype,is_string_dtype
#%%
def infer_best_datetime(s, threshold=0.6):
    """Return kwargs for pd.to_datetime that successfully parses ≥threshold of s."""
    configs = [
        dict(infer_datetime_format=True, dayfirst=False, yearfirst=False),
        dict(infer_datetime_format=True, dayfirst=True),
        dict(infer_datetime_format=True, yearfirst=True),
        dict(format="%Y-%m-%d %H:%M:%S", utc=True),
        dict(format="%Y-%m-%d"),  # added common date format
        dict(format="%m/%d/%Y"),  # added more flexible date format
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
import pandas as pd
import ast
import re

def convert_stringified_lists(df):
    """
    Detects and converts stringified lists in all string columns of the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to process.

    Returns:
        pd.DataFrame: The processed DataFrame with stringified lists converted to actual lists.
    """
    # Regular expression pattern to detect stringified lists
    list_pattern = re.compile(r'^\s*\[.*\]\s*$')

    for col in df.select_dtypes(include=['object', 'string']).columns:
        # Check if any entry in the column matches the list pattern
        if df[col].dropna().apply(lambda x: bool(list_pattern.match(str(x)))).any():
            # Attempt to convert entries to actual lists
            def try_parse(val):
                try:
                    return ast.literal_eval(val) if isinstance(val, str) else val
                except (ValueError, SyntaxError):
                    return val
            df[col] = df[col].apply(try_parse)
    return df

#%%
def normalize_text(val):
    if isinstance(val, list):
        return [normalize_text(v) for v in val]  # recursively clean list elements
    if isinstance(val, str):
        val = val.strip().lower()
        val = re.sub(r'<.*?>', '', val)  # remove HTML
        val = re.sub(r'[^a-z0-9\s]', '', val)  # remove special characters
        return val
    return val

#%%
def normalize_strings(df: pd.DataFrame, logs=None) -> pd.DataFrame:
    if logs is None:
        logs = []

    object_cols = df.select_dtypes(include="object").columns
    for col in object_cols:
        df[col] = df[col].apply(normalize_text)
        logs.append(f"Normalized string values in column '{col}' (lowercased, cleaned HTML, removed special characters).")

    return df
#%%
def preImputationCleaning(dataframe : pd.DataFrame, logs = None) -> pd.DataFrame:
    dataframe = smart_type_inference(dataframe, logs = logs)
    dataframe = normalize_strings(dataframe, logs = logs)
    return dataframe
#%%
def profile_dataframe(df):
    profile_report = []

    for col in df.columns:
        col_data = df[col].dropna()
        col_type = df[col].dtype
        missing_ratio = df[col].isnull().mean()
        entry = {
            'column': col,
            'dtype': col_type,
            'missing_ratio': missing_ratio
        }

        if pd.api.types.is_numeric_dtype(col_type):
            entry['skewness'] = skew(col_data)
            entry['kurtosis'] = kurtosis(col_data)
            entry['correlation_max'] = df.corr(numeric_only=True)[col].drop(col).abs().max()
        elif pd.api.types.is_object_dtype(col_type) or pd.api.types.is_categorical_dtype(col_type):
            num_unique = df[col].nunique()
            entry['cardinality'] = num_unique
            entry['is_categorical_like'] = num_unique < 50

        profile_report.append(entry)

    return pd.DataFrame(profile_report)

#%%
def checkSkewness(data: pd.Series) -> bool:
    return abs(data.skew()) > 1
#%%
def imputeDataForNumericColumn(strategy: str, data: pd.Series, df: pd.DataFrame) -> pd.Series:
    if strategy == 'mean':
        return data.fillna(data.mean())
    elif strategy == 'median':
        return data.fillna(data.median())
    elif strategy == 'mode':
        mode_values = data.mode()
        if not mode_values.empty:
            return data.fillna(mode_values.iloc[0])
        else:
            return data
    elif strategy == 'linear':
        return data.fillna(data.interpolate(method='linear'))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
#%%
def imputeDataForCategoricalData(data : pd.Series) -> pd.Series:
  return data.fillna(data.mode().iloc[0])
#%%
def imputeDataForDateTime(data : pd.Series) -> pd.Series:
   sum_missing = data.isnull().sum()
   sumPercentage = sum_missing / len(data)

   if sumPercentage <= 0.5:
    if isinstance(data.index, pd.DatetimeIndex):
        return data.interpolate(method='time')
    else:
        return data.fillna(method='ffill').fillna(method='bfill')
   else:
        return data.fillna(data.min())
#%%
def defaultDataImputation(data : pd.Series, profile : dict, dataframe : pd.DataFrame) -> pd.Series:
  sum_missing = data.isnull().sum()
  sumPercentage = sum_missing / len(data)

  if is_object_dtype(data) or is_categorical_dtype(data):
    return imputeDataForCategoricalData(data)

  if pd.api.types.is_datetime64_any_dtype(data):
    return imputeDataForDateTime(data)

  if (sumPercentage <= 0.2):
    if (checkSkewness(data)):
      return imputeDataForNumericColumn("median", data,dataframe)
    else:
      return imputeDataForNumericColumn("mean", data,dataframe)
  elif (sumPercentage <= 0.5 and sumPercentage > 0.2):
        return imputeDataForNumericColumn("linear", data,dataframe)
  else:
    return data

#%%
def imputeData(userInstructions : str, data : pd.DataFrame) -> pd.DataFrame:
  if userInstructions is None:
    imputer = HybridImputer(
    missing_ratio_thresh=0.1,
    corr_thresh=0.6,
    min_complete_rows=50,
    cv_splits=5)
    imputer.fit(data)
    return imputer.transform(data)

  for column in data.columns:
    if is_object_dtype(data[column]) or is_categorical_dtype(data[column]) or is_string_dtype(data[column]):
      data[column] = imputeDataForCategoricalData(data[column])
    elif is_numeric_dtype(data[column]):
      data[column] = imputeDataForNumericColumn(userInstructions, data[column], data)
    elif pd.api.types.is_datetime64_any_dtype(data[column]):
      data[column] = imputeDataForDateTime(data[column])
  return data
#%%
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

class HybridImputer:
    def __init__(
        self,
        missing_ratio_thresh: float = 0.1,
        corr_thresh: float = 0.6,
        min_complete_rows: int = 50,
        cv_splits: int = 5,
    ):
        """
        missing_ratio_thresh: below this → rule‑based
        corr_thresh: above this & enough rows → model‑aware
        min_complete_rows: minimum non‑NA rows to train a model
        cv_splits: number of folds for CV when comparing errors
        """
        self.missing_ratio_thresh = missing_ratio_thresh
        self.corr_thresh = corr_thresh
        self.min_complete_rows = min_complete_rows
        self.cv_splits = cv_splits
        self.col_stats_ = {}
        self.strategies_ = {}
        self.models_ = {}

    def fit(self, df: pd.DataFrame):
        # 1) PROFILE
        stats = {}
        # correlations only for numeric
        num_df = df.select_dtypes(include=[np.number])
        corrmat = num_df.corr().abs()
        for col in df.columns:
            col_ser = df[col]
            missing_ratio = col_ser.isna().mean()
            n_unique = col_ser.nunique(dropna=True)
            is_numeric = pd.api.types.is_numeric_dtype(col_ser)
            max_corr = corrmat[col].drop(col).max() if is_numeric and col in corrmat else 0
            complete_rows = col_ser.notna().sum()
            stats[col] = {
                'missing_ratio': missing_ratio,
                'n_unique': n_unique,
                'is_numeric': is_numeric,
                'max_corr': max_corr,
                'complete_rows': complete_rows
            }
        self.col_stats_ = stats

        # 2) DISPATCH: decide rule vs model
        for col, s in stats.items():
            if s['missing_ratio'] < self.missing_ratio_thresh:
                self.strategies_[col] = 'rule'
            elif s['is_numeric'] \
                 and s['max_corr'] > self.corr_thresh \
                 and s['complete_rows'] >= self.min_complete_rows:
                self.strategies_[col] = 'model'
            else:
                self.strategies_[col] = 'rule'

        # 3) TRAIN MODELS for model‑aware columns
        for col, strat in self.strategies_.items():
            if strat == 'model':
                self.models_[col] = self._train_model(df, col)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, strat in self.strategies_.items():
            if strat == 'rule':
                df[col] = defaultDataImputation(df[col],self.col_stats_[col],df)
            else:  # model‑aware with dynamic fallback
                # compare CV errors
                rule_err = self._cv_rule_error(df, col)
                model_err = self._cv_model_error(df, col, self.models_[col])
                if model_err < rule_err:
                    df[col] = self._model_impute(df, col, self.models_[col])
                else:
                    df[col] = defaultDataImputation(df[col],self.col_stats_[col],df)
        return df

    def _train_model(self, df: pd.DataFrame, target_col: str):
        # train on rows where target is present
        not_na = df[target_col].notna()
        X = df.loc[not_na].drop(columns=[target_col])
        y = df.loc[not_na, target_col]
        model = DecisionTreeRegressor()
        model.fit(X, y)
        return model

    def _model_impute(self, df: pd.DataFrame, target_col: str, model) -> pd.Series:
        ser = df[target_col].copy()
        na_idx = ser.isna()
        if na_idx.any():
            X_pred = df.loc[na_idx].drop(columns=[target_col])
            ser.loc[na_idx] = model.predict(X_pred)
        return ser

    def _cv_rule_error(self, df: pd.DataFrame, col: str) -> float:
        """Estimate error of rule-based imputation via CV on complete cases."""
        ser = df[col]
        if ser.isna().sum() == 0:
            return np.inf
        X = df.drop(columns=[col])
        y = ser
        # only use rows that are complete for this col
        complete = y.notna()
        Xc, yc = X.loc[complete], y.loc[complete]
        kf = KFold(n_splits=self.cv_splits, shuffle=True, random_state=0)
        errs = []
        for train_idx, test_idx in kf.split(Xc):
            y_test = yc.iloc[test_idx]
            # rule prediction = mean of training fold
            mean_val = yc.iloc[train_idx].mean()
            errs.append(mean_squared_error(y_test, np.full_like(y_test, mean_val)))
        return np.mean(errs)

    def _cv_model_error(self, df: pd.DataFrame, col: str, model) -> float:
        """Estimate error of model-based imputation via CV on complete cases."""
        ser = df[col]
        if ser.isna().sum() == 0:
            return np.inf
        X = df.drop(columns=[col])
        y = ser
        complete = y.notna()
        Xc, yc = X.loc[complete], y.loc[complete]
        kf = KFold(n_splits=self.cv_splits, shuffle=True, random_state=0)
        errs = []
        for train_idx, test_idx in kf.split(Xc):
            X_train, X_test = Xc.iloc[train_idx], Xc.iloc[test_idx]
            y_train, y_test = yc.iloc[train_idx], yc.iloc[test_idx]
            m = DecisionTreeRegressor()
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
            errs.append(mean_squared_error(y_test, y_pred))
        return np.mean(errs)

#%%
def dataCleanPipeLine(df: pd.DataFrame, userInstructions: str):
    logs = []

    imputationTypes = ["mean","median","mode","linear"]

    if userInstructions is not None:
        userInstructions = userInstructions.strip().lower()

        if userInstructions not in imputationTypes:
            logs.append(f"Unknown imputation type: {userInstructions}")
            logs.append(f"Supported imputation types: {imputationTypes}")
            logs.append(f"Imputation unsuccessful.")
            return df, logs

    original_shape = df.shape
    df = df.drop_duplicates()
    logs.append(f"Removed duplicates: {original_shape[0] - df.shape[0]} rows dropped.")

    df = preImputationCleaning(df, logs=logs)

    df = imputeData(userInstructions, df)

    return df, logs

#%%
df = pd.read_csv('Hotel_Reviews.csv')

df_clean, logs = dataCleanPipeLine(df,None)
