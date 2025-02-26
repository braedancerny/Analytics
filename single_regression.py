import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
from cefpython3 import cefpython as cef
import threading
import sys

class SingleRegressionTab:
    def __init__(self, notebook):
        self.frame = ttk.Frame(notebook)
        self.data = None
        self.show_outliers = False
        self.show_residuals = False
        self.current_x = None
        self.current_y = None
        self.z_threshold = 3.0
        self.fig = None
        self.browser = None

        self.main_frame = ttk.Frame(self.frame)
        self.main_frame.pack(fill="both", expand=True)

        self.left_frame = ttk.Frame(self.main_frame, width=300)
        self.left_frame.pack(side="left", fill="both", expand=False, padx=10, pady=10)

        self.graph_container = ttk.Frame(self.main_frame)
        self.graph_container.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Initialize CEF browser
        self._init_cef()

        ttk.Label(self.left_frame, text="Dependent Variable (Y):").pack(pady=5)
        self.y_dropdown = ttk.Combobox(self.left_frame, state="readonly")
        self.y_dropdown.pack(fill="x", pady=5)
        self._add_tooltip(self.y_dropdown, "Select the variable to predict")

        ttk.Label(self.left_frame, text="Independent Variable (X):").pack(pady=5)
        self.x_dropdown = ttk.Combobox(self.left_frame, state="readonly")
        self.x_dropdown.pack(fill="x", pady=5)
        self._add_tooltip(self.x_dropdown, "Select the predictor variable")

        self.run_button = ttk.Button(self.left_frame, text="Run Regression", command=self.run_regression)
        self.run_button.pack(fill="x", pady=5)
        self._add_tooltip(self.run_button, "Fit the regression model")

        self.outlier_button = ttk.Button(self.left_frame, text="Toggle Outliers", command=self.toggle_outliers)
        self.outlier_button.pack(fill="x", pady=5)
        self._add_tooltip(self.outlier_button, "Show/hide outliers in the plot")

        self.residual_button = ttk.Button(self.left_frame, text="Toggle Residuals", command=self.toggle_residuals)
        self.residual_button.pack(fill="x", pady=5)
        self._add_tooltip(self.residual_button, "Show/hide residual plot")

        self.save_button = ttk.Button(self.left_frame, text="Save Results", command=self.save_results)
        self.save_button.pack(fill="x", pady=5)
        self._add_tooltip(self.save_button, "Save regression results to CSV")

        ttk.Label(self.left_frame, text="Z-score Threshold:").pack(pady=5)
        self.z_entry = ttk.Entry(self.left_frame)
        self.z_entry.insert(0, "3.0")
        self.z_entry.pack(fill="x", pady=5)
        self._add_tooltip(self.z_entry, "Z-score threshold for outlier detection (auto-calculated on run)")

        self.results_box = ttk.LabelFrame(self.left_frame, text="Regression Results", padding=10)
        self.results_box.pack(fill="both", expand=True, pady=10)

        self.result_label = ttk.Label(self.results_box, text="R-squared: ", font=("Arial", 10))
        self.result_label.pack(pady=5)
        self.intercept_label = ttk.Label(self.results_box, text="Intercept: ", font=("Arial", 10))
        self.intercept_label.pack(pady=5)
        self.slope_label = ttk.Label(self.results_box, text="Slope: ", font=("Arial", 10))
        self.slope_label.pack(pady=5)
        self.p_value_label = ttk.Label(self.results_box, text="Slope p-value: ", font=("Arial", 10))
        self.p_value_label.pack(pady=5)
        self.outlier_count_label = ttk.Label(self.results_box, text="Outliers: ", font=("Arial", 10))
        self.outlier_count_label.pack(pady=5)

    def _init_cef(self):
        # Initialize CEF in the main thread
        cef.Initialize()
        sys.excepthook = cef.ExceptHook  # Handle CEF exceptions
        self.browser_frame = cef.CreateBrowserSync(
            url="about:blank",
            window_info=cef.WindowInfo(self.graph_container.winfo_id()),
            settings={"windowless_rendering_enabled": False}
        )
        self.browser_frame.pack(fill="both", expand=True)

    def _add_tooltip(self, widget, text):
        tooltip = tk.Toplevel(widget)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_attributes("-alpha", 0.9)
        label = ttk.Label(
            tooltip, text=text, background="#333333", foreground="white",
            font=("Arial", 8), padding=2, relief="solid", borderwidth=1
        )
        label.pack()
        tooltip.withdraw()

        def show(event):
            x = widget.winfo_rootx() + 10
            y = widget.winfo_rooty() + widget.winfo_height() + 5
            tooltip.wm_geometry(f"+{x}+{y}")
            tooltip.deiconify()
        def hide(event): tooltip.withdraw()
        widget.bind("<Enter>", show)
        widget.bind("<Leave>", hide)
        widget.bind("<FocusOut>", hide)
        widget.bind("<Button-1>", hide)

    def update_dropdowns(self, data: pd.DataFrame) -> None:
        self.data = data
        numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        print(f"SingleRegressionTab - Numerical columns: {numerical_columns}")
        if numerical_columns:
            self.x_dropdown["values"] = numerical_columns
            self.y_dropdown["values"] = numerical_columns
            self.x_dropdown.current(0)
            self.y_dropdown.current(1 if len(numerical_columns) > 1 else 0)

    def calculate_z_threshold(self, X, Y):
        combined_data = np.concatenate([X.flatten(), Y])
        z_scores = np.abs(np.nan_to_num(combined_data - np.mean(combined_data)) / np.std(combined_data))
        return max(2.0, round(np.percentile(z_scores, 95), 1)) if len(z_scores) > 0 else 3.0

    def find_outliers(self, data):
        try:
            self.z_threshold = float(self.z_entry.get())
        except ValueError:
            self.z_threshold = 3.0
        z_scores = np.abs(np.nan_to_num(data - np.mean(data)) / np.std(data))
        return z_scores > self.z_threshold

    def toggle_outliers(self):
        self.show_outliers = not self.show_outliers
        if self.current_x and self.current_y:
            self.run_regression()

    def toggle_residuals(self):
        self.show_residuals = not self.show_residuals
        if self.current_x and self.current_y:
            self.run_regression()

    def run_regression(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded.")
            return

        x_column = self.x_dropdown.get()
        y_column = self.y_dropdown.get()
        if not x_column or not y_column:
            messagebox.showerror("Error", "Please select both X and Y variables.")
            return

        try:
            self.current_x = x_column
            self.current_y = y_column

            X = self.data[[x_column]].values
            Y = self.data[y_column].values
            if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
                messagebox.showerror("Error", "Data contains missing values. Please clean it first.")
                return

            self.z_threshold = self.calculate_z_threshold(X, Y)
            self.z_entry.delete(0, tk.END)
            self.z_entry.insert(0, str(self.z_threshold))

            x_outliers = self.find_outliers(X.flatten())
            y_outliers = self.find_outliers(Y)
            outliers = x_outliers | y_outliers
            outlier_count = np.sum(outliers)
            self.outlier_count_label.config(text=f"Outliers: {outlier_count}")

            df = pd.DataFrame({x_column: X.flatten(), y_column: Y, 'Outlier': outliers})
            X = sm.add_constant(X)
            if self.show_outliers and outlier_count > 0:
                X_clean = X[~outliers]
                Y_clean = Y[~outliers]
                model = sm.OLS(Y_clean, X_clean).fit()
                Y_pred_all = model.predict(X)
            else:
                model = sm.OLS(Y, X).fit()
                Y_pred_all = model.predict(X)

            self.result_label.config(text=f"R-squared: {model.rsquared:.2f}")
            self.intercept_label.config(text=f"Intercept: {model.params[0]:.2f}")
            self.slope_label.config(text=f"Slope: {model.params[1]:.2f}")
            self.p_value_label.config(text=f"Slope p-value: {model.pvalues[1]:.4f}")

            print(f"Plotting data: X={X[:, 1][:5]}, Y={Y[:5]}, Y_pred={Y_pred_all[:5]}")
            if self.show_residuals:
                residuals = Y - Y_pred_all
                self.fig = px.scatter(x=Y_pred_all, y=residuals, 
                                     color=outliers if self.show_outliers else None,
                                     color_discrete_map={True: 'red', False: 'blue'},
                                     title=f"Residuals vs Predicted ({x_column} vs {y_column})",
                                     labels={"x": "Predicted Values", "y": "Residuals"},
                                     hover_data={x_column: X[:, 1], y_column: Y})
                self.fig.add_hline(y=0, line_dash="dash", line_color="red")
            else:
                self.fig = px.scatter(df, x=x_column, y=y_column, 
                                     color='Outlier' if self.show_outliers else None,
                                     color_discrete_map={True: 'red', False: 'blue'},
                                     title=f"{x_column} vs {y_column}{' (Outliers Highlighted)' if self.show_outliers else ''}",
                                     hover_data={'Outlier': True})
                self.fig.add_scatter(x=X[:, 1], y=Y_pred_all, mode='lines', name='Regression Line', line=dict(color='red'))

            html_content = self.fig.to_html(include_plotlyjs=True)
            self.load_plot_in_cef(html_content)

        except Exception as e:
            messagebox.showerror("Error", f"Regression failed: {str(e)}")
            self.load_plot_in_cef(f"<p>Error: {str(e)}</p>")

    def load_plot_in_cef(self, html_content):
        # Save to temp file and load in CEF
        temp_file = "temp_single_regression.html"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        self.browser.LoadUrl(f"file://{os.path.abspath(temp_file)}")

    def clear_plot(self):
        if self.browser:
            self.browser.LoadUrl("about:blank")

    def save_results(self):
        if self.current_x and self.current_y:
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if file_path:
                results = {
                    "R-squared": self.result_label.cget("text").split(": ")[1],
                    "Intercept": self.intercept_label.cget("text").split(": ")[1],
                    "Slope": self.slope_label.cget("text").split(": ")[1],
                    "Slope p-value": self.p_value_label.cget("text").split(": ")[1],
                    "Outliers": self.outlier_count_label.cget("text").split(": ")[1]
                }
                pd.DataFrame([results]).to_csv(file_path, index=False)
                messagebox.showinfo("Success", "Results saved successfully.")