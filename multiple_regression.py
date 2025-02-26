import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import ttkbootstrap as ttk
import plotly.graph_objects as go
from cefpython3 import cefpython as cef
import os

class MultilineTreeview(ttk.Treeview):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bind('<Configure>', self._on_configure)
        style = ttk.Style()
        style.configure('Treeview', rowheight=50, font=("Arial", 10))
        style.configure("Treeview.Heading", font=("Arial", 10, "bold"))

    def _on_configure(self, event):
        self._resize_rows()

    def _resize_rows(self):
        self.update_idletasks()

class MultipleRegressionTab:
    def __init__(self, notebook):
        self.frame = ttk.Frame(notebook)
        self.data = None
        self.stats_cache = {}
        self.current_y = None
        self.current_x = []
        self.last_Y_pred = None
        self.last_X = None
        self.last_Y = None
        self.plot_window = None
        self.browser = None

        self.main_frame = ttk.Frame(self.frame)
        self.main_frame.pack(fill="both", expand=True)
        self.main_frame.columnconfigure(0, weight=3)
        self.main_frame.columnconfigure(1, weight=2)
        self.main_frame.rowconfigure(0, weight=1)

        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.variable_frame = ttk.LabelFrame(self.left_frame, text="Variable Selection", padding=10)
        self.variable_frame.pack(fill="both", expand=False, padx=5, pady=5)

        ttk.Label(self.variable_frame, text="Dependent Variable (Y):").pack(pady=5)
        self.y_dropdown = ttk.Combobox(self.variable_frame, state="readonly")
        self.y_dropdown.pack(fill="x", pady=5)
        self._add_tooltip(self.y_dropdown, "Select the variable to predict")

        ttk.Label(self.variable_frame, text="Independent Variables (X):").pack(pady=5)
        self.x_listbox = tk.Listbox(self.variable_frame, selectmode=tk.MULTIPLE, height=5, font=("Arial", 10))
        self.x_listbox.pack(fill="both", expand=True, pady=5)
        self._add_tooltip(self.x_listbox, "Select variables to use as predictors (Ctrl+click for multiple)")

        self.select_all_button = ttk.Button(self.variable_frame, text="Select All", command=self.select_all)
        self.select_all_button.pack(fill="x", pady=2)
        self._add_tooltip(self.select_all_button, "Select all independent variables")
        self.deselect_all_button = ttk.Button(self.variable_frame, text="Deselect All", command=self.deselect_all)
        self.deselect_all_button.pack(fill="x", pady=2)
        self._add_tooltip(self.deselect_all_button, "Deselect all independent variables")

        self.run_button = ttk.Button(self.variable_frame, text="Run Regression", command=self.run_regression)
        self.run_button.pack(fill="x", pady=5)
        self._add_tooltip(self.run_button, "Fit the regression model")

        self.plot_3d_button = ttk.Button(self.variable_frame, text="Show 3D Plot", command=self.show_3d_plot)
        self.plot_3d_button.pack(fill="x", pady=5)
        self._add_tooltip(self.plot_3d_button, "Open a 3D interactive plot of the regression data")

        self.clear_button = ttk.Button(self.variable_frame, text="Clear Results", command=self.clear_results)
        self.clear_button.pack(fill="x", pady=5)
        self._add_tooltip(self.clear_button, "Reset all results")

        self.results_frame = ttk.LabelFrame(self.left_frame, text="Regression Results", padding=10)
        self.results_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.r_squared_label = ttk.Label(self.results_frame, text="R-squared: ", font=("Arial", 10))
        self.r_squared_label.pack(pady=5, anchor="w")
        self.adj_r_squared_label = ttk.Label(self.results_frame, text="Adjusted R-squared: ", font=("Arial", 10))
        self.adj_r_squared_label.pack(pady=5, anchor="w")
        self.intercept_label = ttk.Label(self.results_frame, text="Intercept: ", font=("Arial", 10))
        self.intercept_label.pack(pady=5, anchor="w")
        self.vif_label = ttk.Label(self.results_frame, text="Max VIF: ", font=("Arial", 10))
        self.vif_label.pack(pady=5, anchor="w")

        self.coefficients_frame = ttk.Frame(self.results_frame)
        self.coefficients_frame.pack(fill="both", expand=True, pady=10)

        self.coefficients_table = MultilineTreeview(
            self.coefficients_frame, columns=("Variable", "Coefficient"), show="headings", height=10
        )
        self._configure_headers(self.coefficients_table, ["Variable", "Coefficient"])
        self.coefficients_table.grid(row=0, column=0, sticky="nsew")
        coeff_v_scroll = ttk.Scrollbar(self.coefficients_frame, orient="vertical", command=self.coefficients_table.yview)
        coeff_h_scroll = ttk.Scrollbar(self.coefficients_frame, orient="horizontal", command=self.coefficients_table.xview)
        self.coefficients_table.configure(yscrollcommand=coeff_v_scroll.set, xscrollcommand=coeff_h_scroll.set)
        coeff_v_scroll.grid(row=0, column=1, sticky="ns")
        coeff_h_scroll.grid(row=1, column=0, sticky="ew")
        self.coefficients_frame.grid_rowconfigure(0, weight=1)
        self.coefficients_frame.grid_columnconfigure(0, weight=1)

        self.stats_frame = ttk.LabelFrame(self.right_frame, text="Descriptive Statistics", padding=10)
        self.stats_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.stats_table_frame = ttk.Frame(self.stats_frame)
        self.stats_table_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.stats_table = MultilineTreeview(
            self.stats_table_frame,
            columns=("Variable", "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"),
            show="headings",
            height=10
        )
        self._configure_headers(self.stats_table, ["Variable", "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"])
        for col in self.stats_table["columns"]:
            self.stats_table.column(col, width=100)
        stats_v_scroll = ttk.Scrollbar(self.stats_table_frame, orient="vertical", command=self.stats_table.yview)
        stats_h_scroll = ttk.Scrollbar(self.stats_table_frame, orient="horizontal", command=self.stats_table.xview)
        self.stats_table.configure(yscrollcommand=stats_v_scroll.set, xscrollcommand=stats_h_scroll.set)
        self.stats_table.grid(row=0, column=0, sticky="nsew")
        stats_v_scroll.grid(row=0, column=1, sticky="ns")
        stats_h_scroll.grid(row=1, column=0, sticky="ew")
        self.stats_table_frame.grid_rowconfigure(0, weight=1)
        self.stats_table_frame.grid_columnconfigure(0, weight=1)

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

    def _configure_headers(self, treeview, columns):
        style = ttk.Style()
        style.configure("Treeview.Heading", font=("Arial", 10, "bold"), anchor="w")
        for col in columns:
            treeview.heading(col, text=col, anchor="w")
            treeview.column(col, anchor="w", stretch=tk.YES)

    def update_dropdowns(self, data: pd.DataFrame) -> None:
        self.data = data
        numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        print(f"MultipleRegressionTab - Numerical columns: {numerical_columns}")
        if numerical_columns:
            self.x_listbox.delete(0, tk.END)
            self.y_dropdown["values"] = numerical_columns
            for col in numerical_columns:
                self.x_listbox.insert(tk.END, col)
            self.y_dropdown.current(0)

    def select_all(self):
        self.x_listbox.select_set(0, tk.END)

    def deselect_all(self):
        self.x_listbox.select_clear(0, tk.END)

    def show_3d_plot(self):
        if self.data is None or self.current_y is None or len(self.current_x) < 2:
            messagebox.showerror("Error", "Run regression with at least two independent variables first.")
            return

        if self.plot_window and self.plot_window.winfo_exists():
            self.plot_window.destroy()

        self.plot_window = ttk.Toplevel(self.frame)
        self.plot_window.title("3D Regression Plot")
        self.plot_window.geometry("800x600")
        self.plot_window.transient(self.frame)
        self.plot_window.grab_set()

        cef.Initialize()
        self.browser = cef.CreateBrowserSync(
            url="about:blank",
            window_info=cef.WindowInfo(self.plot_window.winfo_id()),
            settings={"windowless_rendering_enabled": False}
        )
        self.browser.pack(fill="both", expand=True)

        x1_col, x2_col = self.current_x[:2]
        y_col = self.current_y
        X1 = self.data[x1_col].values
        X2 = self.data[x2_col].values
        Y = self.data[y_col].values
        Y_pred = self.last_Y_pred
        residuals = np.abs(Y - Y_pred)

        df = pd.DataFrame({x1_col: X1, x2_col: X2, y_col: Y, 'Predicted': Y_pred, 'Residual': residuals})
        print(f"3D Plot data: X1={X1[:5]}, X2={X2[:5]}, Y={Y[:5]}, Y_pred={Y_pred[:5]}")
        fig = go.Figure(data=[go.Scatter3d(
            x=df[x1_col], y=df[x2_col], z=df[y_col],
            mode='markers',
            marker=dict(
                size=residuals * 20,
                color=Y_pred,
                colorscale='Viridis',
                opacity=0.6,
                colorbar=dict(title="Predicted Value")
            ),
            hovertemplate=f"{x1_col}: %{{x:.2f}}<br>{x2_col}: %{{y:.2f}}<br>{y_col}: %{{z:.2f}}<br>Predicted: %{{customdata[0]:.2f}}<br>Residual: %{{customdata[1]:.2f}}",
            customdata=np.vstack((Y_pred, residuals)).T
        )])
        fig.update_layout(
            scene=dict(xaxis_title=x1_col, yaxis_title=x2_col, zaxis_title=y_col),
            title="3D Regression Visualization"
        )

        html_content = fig.to_html(include_plotlyjs=True)
        temp_file = "temp_multiple_regression.html"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        self.browser.LoadUrl(f"file://{os.path.abspath(temp_file)}")

    def run_regression(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded.")
            return

        x_columns = [self.x_listbox.get(i) for i in self.x_listbox.curselection()]
        if not x_columns:
            messagebox.showerror("Error", "Please select at least one independent variable.")
            return

        y_column = self.y_dropdown.get()
        if not y_column:
            messagebox.showerror("Error", "Please select a dependent variable.")
            return

        try:
            self.current_x = x_columns
            self.current_y = y_column
            X = self.data[x_columns].values
            Y = self.data[y_column].values
            if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
                messagebox.showerror("Error", "Data contains missing values. Please clean it first.")
                return

            model = LinearRegression()
            model.fit(X, Y)
            Y_pred = model.predict(X)
            self.last_Y_pred = Y_pred
            self.last_X = X
            self.last_Y = Y

            r2 = r2_score(Y, Y_pred)
            n, p = X.shape
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

            vif = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
            max_vif = max(vif) if vif else 0

            self.r_squared_label.config(text=f"R-squared: {r2:.2f}")
            self.adj_r_squared_label.config(text=f"Adjusted R-squared: {adj_r2:.2f}")
            self.intercept_label.config(text=f"Intercept: {model.intercept_:.2f}")
            self.vif_label.config(text=f"Max VIF: {max_vif:.2f}")

            for row in self.coefficients_table.get_children():
                self.coefficients_table.delete(row)
            for i, col in enumerate(x_columns):
                self.coefficients_table.insert("", "end", values=(col, f"{model.coef_[i]:.4f}"))

            self.update_descriptive_stats(x_columns, y_column)

        except Exception as e:
            messagebox.showerror("Error", f"Regression failed: {str(e)}")

    def update_descriptive_stats(self, x_columns: list, y_column: str):
        for row in self.stats_table.get_children():
            self.stats_table.delete(row)
        columns = [y_column] + x_columns
        for col in columns:
            if col not in self.stats_cache:
                stats = self.data[col].describe()
                self.stats_cache[col] = stats
            else:
                stats = self.stats_cache[col]
            self.stats_table.insert("", "end", values=(
                col, f"{stats['count']:.2f}", f"{stats['mean']:.2f}", f"{stats['std']:.2f}",
                f"{stats['min']:.2f}", f"{stats['25%']:.2f}", f"{stats['50%']:.2f}",
                f"{stats['75%']:.2f}", f"{stats['max']:.2f}"
            ))

    def clear_results(self):
        self.r_squared_label.config(text="R-squared: ")
        self.adj_r_squared_label.config(text="Adjusted R-squared: ")
        self.intercept_label.config(text="Intercept: ")
        self.vif_label.config(text="Max VIF: ")
        for row in self.coefficients_table.get_children():
            self.coefficients_table.delete(row)
        for row in self.stats_table.get_children():
            self.stats_table.delete(row)
        self.current_x = []
        self.current_y = None
        self.last_Y_pred = None
        self.last_X = None
        self.last_Y = None
        if self.plot_window and self.plot_window.winfo_exists():
            self.plot_window.destroy()
            self.browser = None