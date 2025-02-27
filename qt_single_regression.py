from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
                             QPushButton, QLineEdit, QScrollArea, QMessageBox, QSizePolicy,
                             QCheckBox)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error

class SingleRegressionTab(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.main_window = parent
        self.data = None
        self.model = None
        self.regression_data = None
        self.outliers_excluded = False
        self.outlier_threshold = 3.0
        self.is_dark_mode = True

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Left panel
        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_frame = QWidget()
        self.left_layout = QVBoxLayout(self.left_frame)
        self.left_layout.setAlignment(Qt.AlignTop)
        self.left_frame.setMinimumWidth(250)

        self.y_label = QLabel("Dependent Variable (Y):")
        self.y_label.setStyleSheet("color: black;")
        self.y_combo = QComboBox()
        self.y_combo.setToolTip("Select the variable to predict")
        self.left_layout.addWidget(self.y_label)
        self.left_layout.addWidget(self.y_combo)

        self.x_label = QLabel("Independent Variable (X):")
        self.x_label.setStyleSheet("color: black;")
        self.x_combo = QComboBox()
        self.x_combo.setToolTip("Select the predictor variable")
        self.left_layout.addWidget(self.x_label)
        self.left_layout.addWidget(self.x_combo)

        self.outlier_check = QCheckBox("Exclude Outliers")
        self.outlier_check.stateChanged.connect(self.toggle_outlier_threshold)
        self.left_layout.addWidget(self.outlier_check)

        self.outlier_threshold_label = QLabel("Z-score Threshold:")
        self.outlier_threshold_label.setStyleSheet("color: black;")
        self.outlier_threshold_entry = QLineEdit("3.0")
        self.outlier_threshold_entry.setEnabled(False)
        self.outlier_threshold_entry.setToolTip("Set the Z-score threshold for outliers")
        self.left_layout.addWidget(self.outlier_threshold_label)
        self.left_layout.addWidget(self.outlier_threshold_entry)

        self.run_button = QPushButton("Run Regression")
        self.run_button.setToolTip("Fit the regression model")
        self.run_button.clicked.connect(self.run_regression)
        self.left_layout.addWidget(self.run_button)

        self.results_frame = QWidget()
        self.results_layout = QVBoxLayout(self.results_frame)
        self.results_frame.setStyleSheet("border: 1px solid #555555; padding: 10px;")
        self.result_label = QLabel("R-squared: ")
        self.result_label.setStyleSheet("color: black;")
        self.intercept_label = QLabel("Intercept: ")
        self.intercept_label.setStyleSheet("color: black;")
        self.slope_label = QLabel("Slope: ")
        self.slope_label.setStyleSheet("color: black;")
        self.p_value_label = QLabel("Slope p-value: ")
        self.p_value_label.setStyleSheet("color: black;")
        self.std_err_label = QLabel("Slope Std Err: ")
        self.std_err_label.setStyleSheet("color: black;")
        self.outlier_count_label = QLabel("Outliers excluded: 0")
        self.outlier_count_label.setStyleSheet("color: black;")
        for label in [self.result_label, self.intercept_label, self.slope_label, 
                      self.p_value_label, self.std_err_label, self.outlier_count_label]:
            self.results_layout.addWidget(label)
        self.left_layout.addWidget(self.results_frame)
        self.left_layout.addStretch(1)

        self.left_scroll.setWidget(self.left_frame)
        self.layout.addWidget(self.left_scroll, stretch=1)

        # Right panel
        self.right_frame = QWidget()
        self.right_layout = QVBoxLayout(self.right_frame)
        self.plot_type_label = QLabel("Plot Type:")
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Regression Plot", "Residual Plot"])
        self.plot_type_combo.currentIndexChanged.connect(self.update_plot)
        self.right_layout.addWidget(self.plot_type_label)
        self.right_layout.addWidget(self.plot_type_combo)

        self.plot_view = QWebEngineView()
        self.plot_view.setHtml("<p>Select variables and run regression.</p>")
        self.plot_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.right_layout.addWidget(self.plot_view)

        self.layout.addWidget(self.right_frame, stretch=3)

    def update_theme(self, is_dark_mode):
        self.is_dark_mode = is_dark_mode

    def update_dropdowns(self, data: pd.DataFrame):
        self.data = data.copy() if data is not None else None
        all_columns = data.columns.tolist() if data is not None else []
        self.x_combo.clear()
        self.y_combo.clear()
        self.x_combo.addItems(all_columns)
        self.y_combo.addItems(all_columns)
        if all_columns:
            self.x_combo.setCurrentIndex(0)
            self.y_combo.setCurrentIndex(1 if len(all_columns) > 1 else 0)

    def toggle_outlier_threshold(self, state):
        self.outlier_threshold_entry.setEnabled(state == Qt.Checked)
        if self.model:
            self.run_regression()

    def run_regression(self):
        if self.main_window.preprocessed_data is None:
            QMessageBox.warning(self, "Warning", "Please preprocess the data first.")
            return
        if self.data is None:
            QMessageBox.critical(self, "Error", "Please load data first.")
            return

        x_column = self.x_combo.currentText()
        y_column = self.y_combo.currentText()
        if not x_column or not y_column:
            QMessageBox.critical(self, "Error", "Select both X and Y variables.")
            return

        if not pd.api.types.is_numeric_dtype(self.data[x_column]) or not pd.api.types.is_numeric_dtype(self.data[y_column]):
            QMessageBox.critical(self, "Error", "Variables must be numeric.")
            return

        try:
            self.outliers_excluded = self.outlier_check.isChecked()
            if self.outliers_excluded:
                self.outlier_threshold = float(self.outlier_threshold_entry.text())
                X = self.data[x_column]
                Y = self.data[y_column]
                X_z = np.abs((X - X.mean()) / X.std())
                Y_z = np.abs((Y - Y.mean()) / Y.std())
                mask = (X_z <= self.outlier_threshold) & (Y_z <= self.outlier_threshold)
                self.regression_data = self.data[mask].copy()
                self.outlier_count_label.setText(f"Outliers excluded: {len(self.data) - len(self.regression_data)}")
            else:
                self.regression_data = self.data.copy()
                self.outlier_count_label.setText("Outliers excluded: 0")

            X = self.regression_data[[x_column]].values
            Y = self.regression_data[y_column].values
            X_with_const = sm.add_constant(X)
            self.model = sm.OLS(Y, X_with_const).fit()

            self.result_label.setText(f"R-squared: {self.model.rsquared:.2f}")
            self.intercept_label.setText(f"Intercept: {self.model.params[0]:.2f}")
            self.slope_label.setText(f"Slope: {self.model.params[1]:.2f}")
            self.p_value_label.setText(f"Slope p-value: {self.model.pvalues[1]:.4f}")
            self.std_err_label.setText(f"Slope Std Err: {self.model.bse[1]:.4f}")

            self.update_plot()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Regression failed: {str(e)}")
            self.plot_view.setHtml(f"<p>Error: {str(e)}</p>")

    def update_plot(self):
        if self.model is None:
            self.plot_view.setHtml("<p>Run regression first.</p>")
            return

        plot_type = self.plot_type_combo.currentText()
        x_column = self.x_combo.currentText()
        y_column = self.y_combo.currentText()

        if plot_type == "Regression Plot":
            fig = px.scatter(self.data, x=x_column, y=y_column, title=f"{x_column} vs {y_column}")
            X_line = np.linspace(self.data[x_column].min(), self.data[x_column].max(), 100)
            Y_line = self.model.params[0] + self.model.params[1] * X_line
            fig.add_scatter(x=X_line, y=Y_line, mode='lines', name='Regression Line', line=dict(color='red'))
            if self.outliers_excluded:
                outliers = self.data[~self.data.index.isin(self.regression_data.index)]
                fig.add_scatter(x=outliers[x_column], y=outliers[y_column], mode='markers', 
                                name='Outliers', marker=dict(color='red'))
            self.plot_view.setHtml(fig.to_html(include_plotlyjs='cdn'))

        elif plot_type == "Residual Plot":
            Y_pred = self.model.predict(sm.add_constant(self.regression_data[x_column].values))
            residuals = self.regression_data[y_column].values - Y_pred
            fig = px.scatter(x=Y_pred, y=residuals, title="Residual Plot", 
                             labels={"x": "Predicted Values", "y": "Residuals"})
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            self.plot_view.setHtml(fig.to_html(include_plotlyjs='cdn'))

    def clear_plot(self):
        self.plot_view.setHtml("<p>Select variables and run regression to see plot.</p>")
        self.result_label.setText("R-squared: ")
        self.intercept_label.setText("Intercept: ")
        self.slope_label.setText("Slope: ")
        self.p_value_label.setText("Slope p-value: ")
        self.std_err_label.setText("Slope Std Err: ")
        self.outlier_count_label.setText("Outliers excluded: 0")
        self.model = None
