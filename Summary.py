#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
def summary(df):
    summary = {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "types_of_data": {},
        "missing_or_unusual": {}
    }

    # Categorize data types
    dtypes = df.dtypes
    for col, dtype in dtypes.items():
        dtype_str = str(dtype)
        summary["types_of_data"].setdefault(dtype_str, []).append(col)

    # Check for missing or unusual values
    for col in df.columns:
        issues = {}
        missing = df[col].isnull().sum()
        if missing > 0:
            issues["missing_values"] = int(missing)

        if df[col].dtype == "object":
            try:
                df[col].astype(float)
                issues["note"] = "contains numeric-looking strings"
            except:
                pass

        if issues:
            summary["missing_or_unusual"][col] = issues

    return summary

#%%
df = pd.read_csv('kz.csv')

print(summary(df))