from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
                             QPushButton, QTreeWidget, QTreeWidgetItem, QMessageBox, 
                             QScrollArea, QSizePolicy)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

class MultipleRegressionTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.data = None
        self.is_dark_mode = True

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

        self.run_button = QPushButton("Run Regression")
        self.run_button.setToolTip("Fit the regression model")
        self.run_button.clicked.connect(self.run_regression)
        self.variable_layout.addWidget(self.run_button)

        self.clear_button = QPushButton("Clear Results")
        self.clear_button.setToolTip("Reset all results")
        self.clear_button.clicked.connect(self.clear_results)
        self.variable_layout.addWidget(self.clear_button)

        self.left_layout.addWidget(self.variable_frame)

        self.results_frame = QWidget()
        self.results_layout = QVBoxLayout(self.results_frame)
        self.results_frame.setStyleSheet("border: 1px solid #555555; padding: 10px;")
        self.r_squared_label = QLabel("R-squared: ")
        self.results_layout.addWidget(self.r_squared_label)
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
        self.plot_view = QWebEngineView()
        self.plot_view.setHtml("<p>Run regression to see plot.</p>")
        self.right_layout.addWidget(self.plot_view)
        self.right_scroll.setWidget(self.right_frame)
        self.layout.addWidget(self.right_scroll, stretch=2)

    def update_theme(self, is_dark_mode):
        self.is_dark_mode = is_dark_mode

    def update_dropdowns(self, data: pd.DataFrame):
        self.data = data.copy() if data is not None else None
        all_columns = data.columns.tolist() if data is not None else []
        self.x_list.clear()
        self.y_combo.clear()
        self.y_combo.addItems(all_columns)
        for col in all_columns:
            item = QTreeWidgetItem([col])
            self.x_list.addTopLevelItem(item)
        if all_columns:
            self.y_combo.setCurrentIndex(0)

    def run_regression(self):
        if self.main_window.preprocessed_data is None:
            QMessageBox.warning(self, "Warning", "Please preprocess the data first.")
            return
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
            X = self.data[x_columns].values
            Y = self.data[y_column].values
            if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
                QMessageBox.critical(self, "Error", "Data contains missing values after preprocessing.")
                return

            model = LinearRegression()
            model.fit(X, Y)
            Y_pred = model.predict(X)

            self.r_squared_label.setText(f"R-squared: {model.score(X, Y):.2f}")
            self.coefficients_tree.clear()
            for i, col in enumerate(x_columns):
                QTreeWidgetItem(self.coefficients_tree, [col, f"{model.coef_[i]:.4f}"])

            if len(x_columns) >= 2:
                fig = px.scatter_3d(self.data, x=x_columns[0], y=x_columns[1], z=y_column, title="3D Regression Plot")
            else:
                fig = px.scatter(self.data, x=x_columns[0], y=y_column, title=f"{x_columns[0]} vs {y_column}")
            self.plot_view.setHtml(fig.to_html(include_plotlyjs='cdn'))

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Regression failed: {str(e)}")

    def clear_results(self):
        self.r_squared_label.setText("R-squared: ")
        self.coefficients_tree.clear()
        self.plot_view.setHtml("<p>Run regression to see plot.</p>")
