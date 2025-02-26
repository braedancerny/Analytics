from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
                             QPushButton, QLineEdit, QScrollArea, QMessageBox, QSizePolicy)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px

class SingleRegressionTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.show_outliers = False
        self.show_residuals = False
        self.current_x = None
        self.current_y = None
        self.z_threshold = 3.0

        self.setStyleSheet("""
            QWidget { background-color: #2b2b2b; }
            QLabel { color: #ffffff; font-size: 14px; }
            QPushButton { background-color: #4a4a4a; color: #ffffff; border: 1px solid #555555; padding: 5px; border-radius: 3px; min-height: 30px; }
            QPushButton:hover { background-color: #666666; }
            QComboBox { background-color: #3c3c3c; color: #ffffff; border: 1px solid #555555; padding: 3px; min-height: 25px; }
            QLineEdit { background-color: #3c3c3c; color: #ffffff; border: 1px solid #555555; padding: 3px; min-height: 25px; }
        """)

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_frame = QWidget()
        self.left_layout = QVBoxLayout(self.left_frame)
        self.left_layout.setAlignment(Qt.AlignTop)
        self.left_frame.setMinimumWidth(250)

        self.y_label = QLabel("Dependent Variable (Y):")
        self.y_combo = QComboBox()
        self.y_combo.setToolTip("Select the variable to predict")
        self.left_layout.addWidget(self.y_label)
        self.left_layout.addWidget(self.y_combo)

        self.x_label = QLabel("Independent Variable (X):")
        self.x_combo = QComboBox()
        self.x_combo.setToolTip("Select the predictor variable")
        self.left_layout.addWidget(self.x_label)
        self.left_layout.addWidget(self.x_combo)

        self.run_button = QPushButton("Run Regression")
        self.run_button.setToolTip("Fit the regression model")
        self.run_button.clicked.connect(self.run_regression)
        self.left_layout.addWidget(self.run_button)

        self.outlier_button = QPushButton("Toggle Outliers")
        self.outlier_button.setToolTip("Show/hide outliers in the plot")
        self.outlier_button.clicked.connect(self.toggle_outliers)
        self.left_layout.addWidget(self.outlier_button)

        self.residual_button = QPushButton("Toggle Residuals")
        self.residual_button.setToolTip("Show/hide residual plot")
        self.residual_button.clicked.connect(self.toggle_residuals)
        self.left_layout.addWidget(self.residual_button)

        self.save_button = QPushButton("Save Results")
        self.save_button.setToolTip("Save regression results to CSV")
        self.save_button.clicked.connect(self.save_results)
        self.left_layout.addWidget(self.save_button)

        self.z_label = QLabel("Z-score Threshold:")
        self.z_entry = QLineEdit("3.0")
        self.z_entry.setToolTip("Z-score threshold for outlier detection (auto-calculated on run)")
        self.left_layout.addWidget(self.z_label)
        self.left_layout.addWidget(self.z_entry)

        self.results_frame = QWidget()
        self.results_layout = QVBoxLayout(self.results_frame)
        self.results_frame.setStyleSheet("border: 1px solid #555555; padding: 10px;")
        self.result_label = QLabel("R-squared: ")
        self.intercept_label = QLabel("Intercept: ")
        self.slope_label = QLabel("Slope: ")
        self.p_value_label = QLabel("Slope p-value: ")
        self.outlier_count_label = QLabel("Outliers: ")
        self.results_layout.addWidget(self.result_label)
        self.results_layout.addWidget(self.intercept_label)
        self.results_layout.addWidget(self.slope_label)
        self.results_layout.addWidget(self.p_value_label)
        self.results_layout.addWidget(self.outlier_count_label)
        self.left_layout.addWidget(self.results_frame)
        self.left_layout.addStretch(1)

        self.left_scroll.setWidget(self.left_frame)
        self.layout.addWidget(self.left_scroll, stretch=1)

        self.plot_view = QWebEngineView()
        self.plot_view.setHtml("<p>Select variables and run regression to see plot.</p>")
        self.plot_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout.addWidget(self.plot_view, stretch=3)

    def update_dropdowns(self, data: pd.DataFrame) -> None:
        self.data = data
        all_columns = data.columns.tolist()
        print("SingleRegressionTab Columns:", all_columns)
        if all_columns:
            self.x_combo.clear()
            self.y_combo.clear()
            self.x_combo.addItems(all_columns)
            self.y_combo.addItems(all_columns)
            self.x_combo.setCurrentIndex(0)
            self.y_combo.setCurrentIndex(1 if len(all_columns) > 1 else 0)

    def calculate_z_threshold(self, X, Y):
        combined_data = np.concatenate([X.flatten(), Y])
        z_scores = np.abs(np.nan_to_num(combined_data - np.mean(combined_data)) / np.std(combined_data))
        return max(2.0, round(np.percentile(z_scores, 95), 1)) if len(z_scores) > 0 else 3.0

    def find_outliers(self, data):
        try:
            self.z_threshold = float(self.z_entry.text())
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
            QMessageBox.critical(self, "Error", "Please load data first.")
            return

        x_column = self.x_combo.currentText()
        y_column = self.y_combo.currentText()
        if not x_column or not y_column:
            QMessageBox.critical(self, "Error", "Please select both X and Y variables.")
            return

        try:
            self.current_x = x_column
            self.current_y = y_column

            X = self.data[[x_column]].values
            Y = self.data[y_column].values
            if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
                QMessageBox.critical(self, "Error", "Data contains missing values. Please clean it first.")
                return

            self.z_threshold = self.calculate_z_threshold(X, Y)
            self.z_entry.setText(str(self.z_threshold))

            x_outliers = self.find_outliers(X.flatten())
            y_outliers = self.find_outliers(Y)
            outliers = x_outliers | y_outliers
            outlier_count = np.sum(outliers)
            self.outlier_count_label.setText(f"Outliers: {outlier_count}")

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

            self.result_label.setText(f"R-squared: {model.rsquared:.2f}")
            self.intercept_label.setText(f"Intercept: {model.params[0]:.2f}")
            self.slope_label.setText(f"Slope: {model.params[1]:.2f}")
            self.p_value_label.setText(f"Slope p-value: {model.pvalues[1]:.4f}")

            if self.show_residuals:
                residuals = Y - Y_pred_all
                fig = px.scatter(x=Y_pred_all, y=residuals, 
                                 color=outliers if self.show_outliers else None,
                                 color_discrete_map={True: 'red', False: 'blue'},
                                 title=f"Residuals vs Predicted ({x_column} vs {y_column})",
                                 labels={"x": "Predicted Values", "y": "Residuals"})
                fig.add_hline(y=0, line_dash="dash", line_color="red")
            else:
                fig = px.scatter(df, x=x_column, y=y_column, 
                                 color='Outlier' if self.show_outliers else None,
                                 color_discrete_map={True: 'red', False: 'blue'},
                                 title=f"{x_column} vs {y_column}{' (Outliers Highlighted)' if self.show_outliers else ''}")
                fig.add_scatter(x=X[:, 1], y=Y_pred_all, mode='lines', name='Regression Line', line=dict(color='red'))

            self.plot_view.setHtml(fig.to_html(include_plotlyjs='cdn'))

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Regression failed: {str(e)}")
            self.plot_view.setHtml(f"<p>Error: {str(e)}</p>")

    def clear_plot(self):
        self.plot_view.setHtml("<p>Select variables and run regression to see plot.</p>")

    def save_results(self):
        if self.current_x and self.current_y:
            from PyQt5.QtWidgets import QFileDialog
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "CSV Files (*.csv)")
            if file_path:
                results = {
                    "R-squared": self.result_label.text().split(": ")[1],
                    "Intercept": self.intercept_label.text().split(": ")[1],
                    "Slope": self.slope_label.text().split(": ")[1],
                    "Slope p-value": self.p_value_label.text().split(": ")[1],
                    "Outliers": self.outlier_count_label.text().split(": ")[1]
                }
                pd.DataFrame([results]).to_csv(file_path, index=False)
                QMessageBox.information(self, "Success", "Results saved successfully.")
