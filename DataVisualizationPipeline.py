#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_diabetes
import os
from typing import List, Optional, Callable
from io import BytesIO
from PIL import Image
from typing import Dict, Any
#%%


class DataVisualizerAssistant:
    def __init__(self, df: pd.DataFrame, output_path: str = "./reports", enable_smart_sampling: bool = True):
        self.original_df = df
        self.output_path = output_path
        self.enable_smart_sampling = enable_smart_sampling

        # Initialize the log storage and plot storage
        self.log_output = []
        self.generated_plots = []

        os.makedirs(self.output_path, exist_ok=True)
        self._log("DataVisualizerAssistant initialized.")

    def _log(self, message: str):
        """Helper method to add logs to the log_output list."""
        self.log_output.append(message)

    def _sample_data(self, df: pd.DataFrame, max_rows: int = 5000, target: str = "") -> pd.DataFrame:
        if not self.enable_smart_sampling or len(df) <= max_rows:
            return df

        self._log(f"Dataset too large ({len(df)} rows). Sampling to {max_rows} rows.")

        if target and target in df.columns:
            try:
                return (
                    df.groupby(target, group_keys=False)
                      .apply(lambda x: x.sample(min(len(x), max_rows // df[target].nunique())))
                      .reset_index(drop=True)
                )
            except Exception as e:
                self._log(f"Stratified sampling failed: {e}")
                return df

        return df.sample(n=max_rows, random_state=42)

    def auto_visualize(self, columns: Optional[List[str]] = None) -> str:
        log_output = ""
        df = self._sample_data(self.original_df)

        if columns is None:
            self._log("[AUTO] Auto-detecting columns...")
            columns = [
                col for col in df.columns
                if df[col].nunique() > 1 and "id" not in col.lower()
            ]
            self._log(f"[AUTO] Using columns: {columns}")

        numeric = df[columns].select_dtypes(include="number")
        categorical = df[columns].select_dtypes(include="object")
        datetime_cols = df[columns].select_dtypes(include="datetime64[ns]")

        try:
            # Correlation heatmap
            if len(numeric.columns) >= 2:
                self._save_plot(
                    lambda: sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm"),
                    "correlation_heatmap.png",
                    "Correlation Heatmap"
                )
            # Pairplot
            if 2 <= len(numeric.columns) <= 5:
                sns.pairplot(numeric.dropna())
                self._finalize_plot("pairplot.png", "Pairplot")

            # Count plots
            for col in categorical.columns:
                if df[col].nunique() <= 10:
                    self._save_plot(
                        lambda: sns.countplot(data=df, x=col),
                        f"countplot_{col}.png",
                        f"Count Plot - {col}",
                        rotate_xticks=True
                    )

            # Distribution plots
            for col in numeric.columns:
                self._save_plot(
                    lambda: sns.kdeplot(data=df, x=col, fill=True),
                    f"distplot_{col}.png",
                    f"Distribution - {col}"
                )

            # Boxplots
            for num_col in numeric.columns:
                for cat_col in categorical.columns:
                    if df[cat_col].nunique() <= 10:
                        self._save_plot(
                            lambda: sns.boxplot(data=df, x=cat_col, y=num_col),
                            f"boxplot_{num_col}_by_{cat_col}.png",
                            f"Boxplot - {num_col} by {cat_col}"
                        )

            # Line plots (time series)
            for time_col in datetime_cols.columns:
                for num_col in numeric.columns:
                    self._save_plot(
                        lambda: sns.lineplot(data=df.sort_values(by=time_col), x=time_col, y=num_col),
                        f"lineplot_{num_col}_over_{time_col}.png",
                        f"Line Plot - {num_col} over {time_col}"
                    )
        except Exception as e:
            self._log(f"Error in auto visualization: {e}")
            log_output += f"Error in auto visualization: {e}\n"

        return log_output

    def custom_plot(self, x: str, y: str = None, kind: str = "scatter", return_fig: bool = False) -> Optional[str | plt.Figure]:
        log_output = ""
        df = self._sample_data(self.original_df, target=y if kind != "hist" else "")

        plot_funcs = {
            "scatter": sns.scatterplot,
            "bar": sns.barplot,
            "box": sns.boxplot,
            "hist": sns.histplot
        }

        if kind not in plot_funcs:
            error_msg = f"Unsupported plot type: {kind}"
            self._log(error_msg)
            log_output += f"{error_msg}\n"
            return log_output

        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_func = plot_funcs[kind]

            if kind == "hist":
                plot_func(data=df, x=x, ax=ax)
            else:
                plot_func(data=df, x=x, y=y, ax=ax)

            title = f"{kind.capitalize()} Plot: {x} vs {y}" if y else f"{kind.capitalize()} Plot: {x}"
            ax.set_title(title)
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()

            if return_fig:
                return fig

            filename = f"{x}_{y}_{kind}.png"
            path = os.path.join(self.output_path, filename)
            fig.savefig(path)
            plt.close(fig)
            self._log(f"Plot saved to {path}")
            log_output += f"Plot saved to {path}\n"
            self.generated_plots.append(path)  # Store the plot path in the list
        except Exception as e:
            self._log(f"Error generating custom plot: {e}")
            log_output += f"Error generating custom plot: {e}\n"

        return log_output

    def convert_to_gradio_image(self, fig: plt.Figure) -> Image.Image:
        try:
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            plt.close(fig)
            return Image.open(buf)
        except Exception as e:
            self._log(f"Error converting figure to Gradio image: {e}")
            return None

    def _save_plot(self, plot_func: Callable, filename: str, title: str, rotate_xticks: bool = False):
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_func()
            ax.set_title(title)
            if rotate_xticks:
                plt.xticks(rotation=45)
            self._finalize_plot(filename, title)
            plt.close(fig)
        except Exception as e:
            self._log(f"Error in saving plot: {e}")

    def _finalize_plot(self, filename: str, title: str):
        try:
            path = os.path.join(self.output_path, filename)
            plt.tight_layout()
            plt.savefig(path)
            self._log(f"{title} saved to {path}")
            self.generated_plots.append(path)  # Store the plot path in the list
        except Exception as e:
            self._log(f"Error finalizing plot {filename}: {e}")

    def get_logs_and_plots(self) -> dict:
        """Returns the logs and the generated plots."""
        return {
            "logs": self.log_output,
            "plots": self.generated_plots
        }

#%%
#FOR GRADIO INTEGRATION

# from typing import List, Optional, Callable, Dict, Any
# from io import BytesIO
# from PIL import Image


# class DataVisualizerAssistant:
#     def __init__(
#         self,
#         df: pd.DataFrame,
#         enable_smart_sampling: bool = True,
#         sample_size: int = 5000
#     ):
#         self.original_df      = df
#         self.enable_smart_sampling = enable_smart_sampling
#         self.sample_size      = sample_size

#         # storage for logs and generated images
#         self.log_output       : List[str]   = []
#         self.generated_plots  : List[Image.Image] = []

#         self._log("DataVisualizerAssistant initialized.")

#     def _log(self, message: str):
#         """Append a message to the internal log."""
#         self.log_output.append(message)

#     def _sample_data(self, df: pd.DataFrame, target: str = "") -> pd.DataFrame:
#         if not self.enable_smart_sampling or len(df) <= self.sample_size:
#             return df

#         self._log(f"[INFO] Sampling down from {len(df)} to {self.sample_size} rows.")
#         if target and target in df.columns:
#             try:
#                 sampled = (
#                     df.groupby(target, group_keys=False)
#                       .apply(lambda grp: grp.sample(
#                           min(len(grp), self.sample_size // df[target].nunique()),
#                           random_state=42
#                       ))
#                       .reset_index(drop=True)
#                 )
#                 self._log("[INFO] Stratified sampling successful.")
#                 return sampled
#             except Exception as e:
#                 self._log(f"[WARN] Stratified sampling failed: {e}")

#         return df.sample(n=self.sample_size, random_state=42)

#     def _fig_to_image(self, fig: plt.Figure) -> Image.Image:
#         buf = BytesIO()
#         fig.savefig(buf, format="png", bbox_inches="tight")
#         buf.seek(0)
#         plt.close(fig)
#         return Image.open(buf)

#     def auto_visualize(self, columns: Optional[List[str]] = None) -> Dict[str, Any]:
#         """
#         Automatically generate a suite of plots over your data, excluding any column with 'id'.
#         Returns a dict with keys:
#           - "logs": List[str]
#           - "plots": List[PIL.Image]
#         """
#         df = self._sample_data(self.original_df)

#         # pick relevant columns if none provided
#         if columns is None:
#             self._log("[AUTO] Detecting columns (dropping any with 'id' or nunique≤1)...")
#             columns = [
#                 c for c in df.columns
#                 if df[c].nunique() > 1 and "id" not in c.lower()
#             ]
#             self._log(f"[AUTO] Columns: {columns}")

#         num = df[columns].select_dtypes(include="number")
#         cat = df[columns].select_dtypes(include="object")
#         dt  = df[columns].select_dtypes(include="datetime64[ns]")

#         # 1) Correlation heatmap
#         if len(num.columns) >= 2:
#             fig, ax = plt.subplots(figsize=(8,6))
#             sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
#             ax.set_title("Correlation Heatmap")
#             img = self._fig_to_image(fig)
#             self.generated_plots.append(img)
#             self._log("[AUTO] Correlation heatmap generated.")

#         # 2) Pairplot (2–5 numeric cols)
#         if 2 <= len(num.columns) <= 5:
#             grid = sns.pairplot(num.dropna())
#             img = self._fig_to_image(grid.fig)
#             self.generated_plots.append(img)
#             self._log("[AUTO] Pairplot generated.")

#         # 3) Count plots for low‑card cat cols
#         for c in cat.columns:
#             if df[c].nunique() <= 10:
#                 fig, ax = plt.subplots(figsize=(6,4))
#                 sns.countplot(data=df, x=c, ax=ax)
#                 ax.set_title(f"Count Plot — {c}")
#                 plt.setp(ax.get_xticklabels(), rotation=45)
#                 img = self._fig_to_image(fig)
#                 self.generated_plots.append(img)
#                 self._log(f"[AUTO] Count plot for '{c}' generated.")

#         # 4) Distribution (KDE) for numeric cols
#         for c in num.columns:
#             fig, ax = plt.subplots(figsize=(6,4))
#             sns.kdeplot(data=df, x=c, fill=True, ax=ax)
#             ax.set_title(f"Distribution — {c}")
#             img = self._fig_to_image(fig)
#             self.generated_plots.append(img)
#             self._log(f"[AUTO] KDE distribution for '{c}' generated.")

#         # 5) Boxplots: each num vs each low‑card cat
#         for ncol in num.columns:
#             for ccol in cat.columns:
#                 if df[ccol].nunique() <= 10:
#                     fig, ax = plt.subplots(figsize=(6,4))
#                     sns.boxplot(data=df, x=ccol, y=ncol, ax=ax)
#                     ax.set_title(f"Boxplot — {ncol} by {ccol}")
#                     img = self._fig_to_image(fig)
#                     self.generated_plots.append(img)
#                     self._log(f"[AUTO] Boxplot '{ncol}'|'{ccol}' generated.")

#         # 6) Line plots for time series
#         for dcol in dt.columns:
#             for ncol in num.columns:
#                 fig, ax = plt.subplots(figsize=(6,4))
#                 sns.lineplot(data=df.sort_values(by=dcol), x=dcol, y=ncol, ax=ax)
#                 ax.set_title(f"LinePlot — {ncol} over {dcol}")
#                 img = self._fig_to_image(fig)
#                 self.generated_plots.append(img)
#                 self._log(f"[AUTO] Time‑series '{ncol}' over '{dcol}' generated.")

#         return {
#             "logs":  self.log_output.copy(),
#             "plots": self.generated_plots.copy()
#         }

#     def custom_plot(
#         self,
#         x: str,
#         y: Optional[str] = None,
#         kind: str = "scatter"
#     ) -> Dict[str, Any]:
#         """
#         Generate a single plot of type `kind` and return it + any logs.
#         kind ∈ {"scatter","bar","box","hist"}.
#         """
#         df = self._sample_data(self.original_df, target=y if kind!="hist" else "")
#         funcs = {
#             "scatter": sns.scatterplot,
#             "bar":     sns.barplot,
#             "box":     sns.boxplot,
#             "hist":    sns.histplot
#         }

#         if kind not in funcs:
#             msg = f"[ERROR] Unsupported plot type: {kind}"
#             self._log(msg)
#             return {"logs": self.log_output.copy(), "plots": []}

#         try:
#             fig, ax = plt.subplots(figsize=(6,4))
#             if kind=="hist":
#                 funcs[kind](data=df, x=x, ax=ax)
#             else:
#                 funcs[kind](data=df, x=x, y=y, ax=ax)

#             ax.set_title(f"{kind.capitalize()} — {x}" + (f" vs {y}" if y else ""))
#             plt.setp(ax.get_xticklabels(), rotation=45)

#             img = self._fig_to_image(fig)
#             self.generated_plots.append(img)
#             self._log(f"[CUSTOM] {kind} plot ({x}{','+y if y else ''}) generated.")

#             return {"logs": self.log_output.copy(), "plots": [img]}

#         except Exception as e:
#             self._log(f"[ERROR] Custom plot failed: {e}")
#             return {"logs": self.log_output.copy(), "plots": []}

#%%
def dataVisualizationPipeline(
    df: pd.DataFrame,
    column1: str = None,
    column2: str = None,
    pltType: str = None
) -> Dict[str, Any]:
    """
    A one‑stop pipeline that:
      - If pltType is None OR column1 is None: runs full auto‑EDA.
      - Otherwise: produces exactly one custom plot of type `pltType` on (column1[,column2]).

    Returns a dict with:
      - "logs": List[str]
      - "plots": List[PIL.Image]
    """
    viz = DataVisualizerAssistant(df)

    # full auto‑EDA if no plot type or no primary column specified
    if pltType is None or column1 is None:
        viz.auto_visualize()
    else:
        # custom plot: kind must be one of ["scatter","bar","box","hist"]
        viz.custom_plot(x=column1, y=column2, kind=pltType)

    # gather everything the assistant has accumulated
    return {
        "logs":  viz.log_output.copy(),
        "plots": viz.generated_plots.copy()
    }

#%%
from sklearn.datasets import load_diabetes


# 1. Load and prepare DataFrame
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# 2. Instantiate (disable sampling so nothing is dropped)
viz = DataVisualizerAssistant(df, enable_smart_sampling=False)

# 3a. Auto‑visualize
auto_logs = viz.auto_visualize()
print("=== auto_visualize logs ===")
print(auto_logs or "(no errors)")

# 3b. Custom histogram of 'bmi'
hist_logs = viz.custom_plot(x='bmi', kind='hist')
print("\n=== custom_plot(hist) logs ===")
print(hist_logs or "(no errors)")

# 3c. Custom scatter of 'bmi' vs. 'target'
scatter_logs = viz.custom_plot(x='bmi', y='target', kind='scatter')
print("\n=== custom_plot(scatter) logs ===")
print(scatter_logs or "(no errors)")

# 4. Finally, inspect all logs and list of generated plot file paths
results = viz.get_logs_and_plots()
print("\n=== All accumulated logs ===")
for line in results['logs']:
    print("  ", line)

print("\n=== Generated plot files ===")
for path in results['plots']:
    print("  ", path)
