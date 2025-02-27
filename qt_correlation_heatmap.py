from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem, 
                             QComboBox, QPushButton, QMessageBox)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt
import pandas as pd
import plotly.express as px

class CorrelationHeatmapTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.data = None
        self.is_dark_mode = True

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.left_frame = QWidget()
        self.left_layout = QVBoxLayout(self.left_frame)
        self.var_list = QTreeWidget()
        self.var_list.setHeaderHidden(True)
        self.var_list.setSelectionMode(QTreeWidget.MultiSelection)
        self.var_list.setToolTip("Select variables for correlation (Ctrl+click for multiple)")
        self.method_combo = QComboBox()
        self.method_combo.addItems(["pearson", "spearman"])
        self.method_combo.setToolTip("Select correlation method")
        self.run_button = QPushButton("Generate Heatmap")
        self.run_button.setToolTip("Generate the correlation heatmap")
        self.run_button.clicked.connect(self.generate_heatmap)
        self.left_layout.addWidget(self.var_list)
        self.left_layout.addWidget(self.method_combo)
        self.left_layout.addWidget(self.run_button)

        self.plot_view = QWebEngineView()
        self.plot_view.setHtml("<p>Select variables and generate heatmap.</p>")
        self.layout.addWidget(self.left_frame, stretch=1)
        self.layout.addWidget(self.plot_view, stretch=3)

    def update_theme(self, is_dark_mode):
        self.is_dark_mode = is_dark_mode

    def update_dropdowns(self, data: pd.DataFrame):
        self.data = data.copy() if data is not None else None
        all_columns = data.columns.tolist() if data is not None else []
        self.var_list.clear()
        if all_columns:
            for col in all_columns:
                item = QTreeWidgetItem([col])
                self.var_list.addTopLevelItem(item)

    def generate_heatmap(self):
        if self.main_window.preprocessed_data is None:
            QMessageBox.warning(self, "Warning", "Please preprocess the data first.")
            return
        if self.data is None:
            QMessageBox.critical(self, "Error", "No data loaded.")
            return
        selected_items = self.var_list.selectedItems()
        if len(selected_items) < 2:
            QMessageBox.critical(self, "Error", "Select at least two variables.")
            return
        selected_vars = [item.text(0) for item in selected_items]
        method = self.method_combo.currentText()
        corr_matrix = self.data[selected_vars].corr(method=method)
        fig = px.imshow(corr_matrix, 
                        text_auto=True, 
                        color_continuous_scale="RdBu_r",
                        color_continuous_midpoint=0,
                        zmin=-1,
                        zmax=1,
                        title=f"Correlation Heatmap ({method.capitalize()})")
        self.plot_view.setHtml(fig.to_html(include_plotlyjs="cdn"))
