from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
                             QPushButton, QTreeWidget, QTreeWidgetItem, QMessageBox, 
                             QMainWindow, QScrollArea, QSizePolicy, QCheckBox, QLineEdit)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.graph_objects as go
import plotly.io as pio

class MultipleRegressionTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.stats_cache = {}
        self.current_y = None
        self.current_x = []
        self.last_Y_pred = None
        self.last_X = None
        self.last_Y = None
        self.plot_window = None
        self.is_dark_mode = True

        # Set all labels to black
        self.setStyleSheet("QLabel { color: #000000; }")

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_frame = QWidget()
        self.left_layout = QVBoxLayout(self.left_frame)
        self.left_layout.setAlignment(Qt.AlignTop)
        self.left_frame.setMinimumWidth(250)

        self.variable_frame = QWidget()
        self.variable_layout = QVBoxLayout(self.variable_frame)
        self.variable_frame.setStyleSheet("border: 1px solid #555555; padding: 10px;")
        self.y_label = QLabel("Dependent Variable (Y):")
        self.y_combo = QComboBox()
        self.y_combo.setToolTip("Select the variable to predict")
        self.variable_layout.addWidget(self.y_label)
        self.variable_layout.addWidget(self.y_combo)

        self.x_label = QLabel("Independent Variables (X):")
        self.x_list = QTreeWidget()
        self.x_list.setHeaderHidden(True)
        self.x_list.setSelectionMode(QTreeWidget.MultiSelection)
        self.x_list.setToolTip("Select variables to use as predictors (Ctrl+click for multiple)")
        self.variable_layout.addWidget(self.x_label)
        self.variable_layout.addWidget(self.x_list)

        self.select_all_button = QPushButton("Select All")
        self.select_all_button.setToolTip("Select all independent variables")
        self.select_all_button.clicked.connect(self.select_all)
        self.deselect_all_button = QPushButton("Deselect All")
        self.deselect_all_button.setToolTip("Deselect all independent variables")
        self.deselect_all_button.clicked.connect(self.deselect_all)
        self.feature_selection_button = QPushButton("Select Features (RFE)")
        self.feature_selection_button.setToolTip("Automatically select top features using RFE")
        self.feature_selection_button.clicked.connect(self.select_features)
        self.variable_layout.addWidget(self.select_all_button)
        self.variable_layout.addWidget(self.deselect_all_button)
        self.variable_layout.addWidget(self.feature_selection_button)

        self.run_button = QPushButton("Run Regression")
        self.run_button.setToolTip("Fit the regression model")
        self.run_button.clicked.connect(self.run_regression)
        self.variable_layout.addWidget(self.run_button)

        self.plot_3d_button = QPushButton("Show 3D Plot")
        self.plot_3d_button.setToolTip("Open a 3D interactive plot of the regression data")
        self.plot_3d_button.clicked.connect(self.show_3d_plot)
        self.variable_layout.addWidget(self.plot_3d_button)

        self.clear_button = QPushButton("Clear Results")
        self.clear_button.setToolTip("Reset all results")
        self.clear_button.clicked.connect(self.clear_results)
        self.variable_layout.addWidget(self.clear_button)

        self.cv_check = QCheckBox("Enable Cross-Validation")
        self.cv_check.setToolTip("Perform k-fold cross-validation")
        self.cv_folds_entry = QLineEdit("5")
        self.cv_folds_entry.setToolTip("Number of folds for cross-validation")
        self.variable_layout.addWidget(self.cv_check)
        self.variable_layout.addWidget(self.cv_folds_entry)

        self.left_layout.addWidget(self.variable_frame)

        self.results_frame = QWidget()
        self.results_layout = QVBoxLayout(self.results_frame)
        self.results_frame.setStyleSheet("border: 1px solid #555555; padding: 10px;")
        self.r_squared_label = QLabel("R-squared: ")
        self.adj_r_squared_label = QLabel("Adjusted R-squared: ")
        self.intercept_label = QLabel("Intercept: ")
        self.vif_label = QLabel("Max VIF: ")
        self.mae_label = QLabel("MAE: ")
        self.mse_label = QLabel("MSE: ")
        self.rmse_label = QLabel("RMSE: ")
        self.cv_r2_label = QLabel("CV R-squared: ")
        for label in [self.r_squared_label, self.adj_r_squared_label, self.intercept_label, 
                      self.vif_label, self.mae_label, self.mse_label, self.rmse_label, self.cv_r2_label]:
            self.results_layout.addWidget(label)

        self.coefficients_tree = QTreeWidget()
        self.coefficients_tree.setHeaderLabels(["Variable", "Coefficient"])
        self.coefficients_tree.setColumnWidth(0, 150)
        self.results_layout.addWidget(self.coefficients_tree)
        self.left_layout.addWidget(self.results_frame)
        self.left_layout.addStretch(1)

        self.left_scroll.setWidget(self.left_frame)
        self.layout.addWidget(self.left_scroll, stretch=1)

        self.right_scroll = QScrollArea()
        self.right_scroll.setWidgetResizable(True)
        self.right_frame = QWidget()
        self.right_layout = QVBoxLayout(self.right_frame)
        self.right_layout.setAlignment(Qt.AlignTop)
        self.stats_frame = QWidget()
        self.stats_layout = QVBoxLayout(self.stats_frame)
        self.stats_frame.setStyleSheet("border: 1px solid #555555; padding: 10px;")
        self.stats_tree = QTreeWidget()
        self.stats_tree.setHeaderLabels(["Variable", "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"])
        for i in range(self.stats_tree.columnCount()):
            self.stats_tree.setColumnWidth(i, 100)
        self.stats_tree.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.stats_layout.addWidget(self.stats_tree)
        self.right_layout.addWidget(self.stats_frame)
        self.right_layout.addStretch(1)
        self.right_scroll.setWidget(self.right_frame)
        self.layout.addWidget(self.right_scroll, stretch=2)

    def update_theme(self, is_dark_mode):
        self.is_dark_mode = is_dark_mode
        # No need for specific style updates; global stylesheet handles it

    def update_dropdowns(self, data: pd.DataFrame) -> None:
        self.data = data
        all_columns = data.columns.tolist() if data is not None else []
        self.x_list.clear()
        self.y_combo.clear()
        self.y_combo.addItems(all_columns)
        for col in all_columns:
            item = QTreeWidgetItem([col])
            self.x_list.addTopLevelItem(item)
        if all_columns:
            self.y_combo.setCurrentIndex(0)

    def select_all(self):
        for i in range(self.x_list.topLevelItemCount()):
            self.x_list.topLevelItem(i).setSelected(True)

    def deselect_all(self):
        for i in range(self.x_list.topLevelItemCount()):
            self.x_list.topLevelItem(i).setSelected(False)

    def select_features(self):
        if self.data is None or self.y_combo.currentText() == "":
            QMessageBox.critical(self, "Error", "Please load data and select a dependent variable.")
            return
        x_columns = [self.x_list.topLevelItem(i).text(0) for i in range(self.x_list.topLevelItemCount())]
        y_column = self.y_combo.currentText()
        X = self.data[x_columns].values
        Y = self.data[y_column].values
        if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
            QMessageBox.critical(self, "Error", "Data contains missing values.")
            return
        model = LinearRegression()
        rfe = RFE(model, n_features_to_select=min(3, len(x_columns)))
        rfe.fit(X, Y)
        selected_features = [x_columns[i] for i in range(len(rfe.support_)) if rfe.support_[i]]
        for i in range(self.x_list.topLevelItemCount()):
            item = self.x_list.topLevelItem(i)
            item.setSelected(item.text(0) in selected_features)

    def show_3d_plot(self):
        if self.data is None or self.current_y is None or len(self.current_x) < 2:
            QMessageBox.critical(self, "Error", "Run regression with at least two independent variables first.")
            return

        if self.plot_window and not self.plot_window.isHidden():
            self.plot_window.close()

        self.plot_window = QMainWindow(self)
        self.plot_window.setWindowTitle("3D Regression Plot")
        self.plot_window.setGeometry(200, 200, 800, 600)

        central_widget = QWidget()
        self.plot_window.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        plot_view = QWebEngineView()
        layout.addWidget(plot_view)

        x1_col, x2_col = self.current_x[:2]
        y_col = self.current_y
        X1 = self.data[x1_col].values
        X2 = self.data[x2_col].values
        Y = self.data[y_col].values
        Y_pred = self.last_Y_pred
        residuals = np.abs(Y - Y_pred)

        valid_mask = ~np.isnan(X1) & ~np.isnan(X2) & ~np.isnan(Y) & ~np.isnan(Y_pred) & ~np.isnan(residuals)
        X1_clean = X1[valid_mask]
        X2_clean = X2[valid_mask]
        Y_clean = Y[valid_mask]
        Y_pred_clean = Y_pred[valid_mask]
        residuals_clean = residuals[valid_mask]

        if len(X1_clean) == 0:
            QMessageBox.critical(self, "Error", "No valid data points to plot after removing NaN values.")
            return

        df = pd.DataFrame({
            x1_col: X1_clean,
            x2_col: X2_clean,
            y_col: Y_clean,
            'Predicted': Y_pred_clean,
            'Residual': residuals_clean
        })

        min_size, max_size = 5, 20
        residual_min, residual_max = np.min(residuals_clean), np.max(residuals_clean)
        if residual_max == residual_min:
            sizes = np.full_like(residuals_clean, min_size)
        else:
            sizes = min_size + (max_size - min_size) * (residuals_clean - residual_min) / (residual_max - residual_min)

        fig = go.Figure(data=[go.Scatter3d(
            x=df[x1_col], y=df[x2_col], z=df[y_col],
            mode='markers',
            marker=dict(
                size=sizes,
                color=Y_pred_clean,
                colorscale='Viridis',
                opacity=0.6,
                colorbar=dict(title="Predicted Value")
            )
        )])
        fig.update_layout(
            scene=dict(xaxis_title=x1_col, yaxis_title=x2_col, zaxis_title=y_col),
            title="3D Regression Visualization"
        )
        plot_view.setHtml(fig.to_html(include_plotlyjs='cdn'))
        self.plot_window.show()

    def run_regression(self):
        if self.data is None:
            QMessageBox.critical(self, "Error", "No data loaded.")
            return

        x_items = self.x_list.selectedItems()
        x_columns = [item.text(0) for item in x_items]
        if not x_columns:
            QMessageBox.critical(self, "Error", "Please select at least one independent variable.")
            return

        y_column = self.y_combo.currentText()
        if not y_column:
            QMessageBox.critical(self, "Error", "Please select a dependent variable.")
            return

        try:
            self.current_x = x_columns
            self.current_y = y_column
            X = self.data[x_columns].values
            Y = self.data[y_column].values
            if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
                QMessageBox.critical(self, "Error", "Data contains missing values.")
                return

            if X.size == 0 or X.shape[1] == 0:
                QMessageBox.critical(self, "Error", "No valid data selected.")
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
            vif = [variance_inflation_factor(X, i) for i in range(X.shape[1])] if X.shape[1] > 1 else [0]
            max_vif = max(vif) if vif else 0

            self.r_squared_label.setText(f"R-squared: {r2:.2f}")
            self.adj_r_squared_label.setText(f"Adjusted R-squared: {adj_r2:.2f}")
            self.intercept_label.setText(f"Intercept: {model.intercept_:.2f}")
            self.vif_label.setText(f"Max VIF: {max_vif:.2f}")
            self.mae_label.setText(f"MAE: {mean_absolute_error(Y, Y_pred):.2f}")
            self.mse_label.setText(f"MSE: {mean_squared_error(Y, Y_pred):.2f}")
            self.rmse_label.setText(f"RMSE: {np.sqrt(mean_squared_error(Y, Y_pred)):.2f}")

            if self.cv_check.isChecked():
                folds = int(self.cv_folds_entry.text())
                cv_scores = cross_val_score(model, X, Y, cv=folds, scoring='r2')
                self.cv_r2_label.setText(f"CV R-squared: {np.mean(cv_scores):.2f}")
            else:
                self.cv_r2_label.setText("CV R-squared: ")

            self.coefficients_tree.clear()
            for i, col in enumerate(x_columns):
                QTreeWidgetItem(self.coefficients_tree, [col, f"{model.coef_[i]:.4f}"])

            self.update_descriptive_stats(x_columns, y_column)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Regression failed: {str(e)}")

    def update_descriptive_stats(self, x_columns: list, y_column: str):
        self.stats_tree.clear()
        columns = [y_column] + x_columns
        for col in columns:
            if col not in self.stats_cache:
                stats = self.data[col].describe()
                self.stats_cache[col] = stats
            else:
                stats = self.stats_cache[col]
            item = QTreeWidgetItem([
                col, f"{stats['count']:.2f}", f"{stats['mean']:.2f}", f"{stats['std']:.2f}",
                f"{stats['min']:.2f}", f"{stats['25%']:.2f}", f"{stats['50%']:.2f}",
                f"{stats['75%']:.2f}", f"{stats['max']:.2f}"
            ])
            self.stats_tree.addTopLevelItem(item)

    def clear_results(self):
        self.r_squared_label.setText("R-squared: ")
        self.adj_r_squared_label.setText("Adjusted R-squared: ")
        self.intercept_label.setText("Intercept: ")
        self.vif_label.setText("Max VIF: ")
        self.mae_label.setText("MAE: ")
        self.mse_label.setText("MSE: ")
        self.rmse_label.setText("RMSE: ")
        self.cv_r2_label.setText("CV R-squared: ")
        self.coefficients_tree.clear()
        self.stats_tree.clear()
        self.current_x = []
        self.current_y = None
        self.last_Y_pred = None
        self.last_X = None
        self.last_Y = None
        if self.plot_window and not self.plot_window.isHidden():
            self.plot_window.close()
