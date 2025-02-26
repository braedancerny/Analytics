from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
                             QPushButton, QTreeWidget, QTreeWidgetItem, QMessageBox,
                             QMainWindow)
from PyQt5.QtCore import Qt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataPreprocessingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.is_dark_mode = True

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.missing_label = QLabel("Handle Missing Values:")
        self.missing_combo = QComboBox()
        self.missing_combo.addItems(["Fill with Mean", "Fill with Median", "Fill with Zero", "Drop Rows"])
        self.missing_combo.setToolTip("Select method to handle missing values")

        self.scaling_label = QLabel("Scale Features:")
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems(["None", "Standard Scaler", "Min-Max Scaler"])
        self.scaling_combo.setToolTip("Choose feature scaling method")

        self.update_label_styles()
        self.layout.addWidget(self.missing_label)
        self.layout.addWidget(self.missing_combo)
        self.layout.addWidget(self.scaling_label)
        self.layout.addWidget(self.scaling_combo)

        self.apply_button = QPushButton("Apply Preprocessing")
        self.apply_button.setToolTip("Apply selected preprocessing steps")
        self.apply_button.clicked.connect(self.apply_preprocessing)
        self.layout.addWidget(self.apply_button)

        self.data_viewer = QTreeWidget()
        self.data_viewer.setHeaderHidden(False)
        self.layout.addWidget(self.data_viewer, stretch=1)

    def update_label_styles(self):
        self.missing_label.setStyleSheet("color: #000000;")
        self.scaling_label.setStyleSheet("color: #000000;")

    def update_theme(self, is_dark_mode):
        self.is_dark_mode = is_dark_mode
        self.update_label_styles()

    def update_dropdowns(self, data: pd.DataFrame):
        self.data = data.copy() if data is not None else None
        self._populate_table(self.data)

    def _populate_table(self, data: pd.DataFrame):
        self.data_viewer.clear()
        if data is None or data.empty:
            return
        self.data_viewer.setColumnCount(len(data.columns))
        self.data_viewer.setHeaderLabels(list(data.columns))
        for col in range(len(data.columns)):
            self.data_viewer.setColumnWidth(col, 100)
        for _, row in data.iterrows():
            item = QTreeWidgetItem([str(val) if pd.notnull(val) else 'NaN' for val in row])
            self.data_viewer.addTopLevelItem(item)
        self.data_viewer.viewport().update()

    def apply_preprocessing(self):
        global data
        if self.data is None:
            QMessageBox.critical(self, "Error", "No data loaded.")
            return
        
        processed_data = self.data.copy()
        print("Before handling NaN:\n", processed_data.to_string())

        # Handle missing values
        missing_method = self.missing_combo.currentText()
        if missing_method == "Fill with Mean":
            processed_data.fillna(processed_data.mean(), inplace=True)
        elif missing_method == "Fill with Median":
            processed_data.fillna(processed_data.median(), inplace=True)
        elif missing_method == "Fill with Zero":
            processed_data.fillna(0, inplace=True)
        elif missing_method == "Drop Rows":
            processed_data.dropna(inplace=True)

        # Apply scaling
        scaling_method = self.scaling_combo.currentText()
        numeric_cols = processed_data.select_dtypes(include=['float64', 'int64']).columns
        if scaling_method == "Standard Scaler":
            scaler = StandardScaler()
            processed_data[numeric_cols] = scaler.fit_transform(processed_data[numeric_cols])
        elif scaling_method == "Min-Max Scaler":
            scaler = MinMaxScaler()
            processed_data[numeric_cols] = scaler.fit_transform(processed_data[numeric_cols])

        print("After handling NaN and scaling:\n", processed_data.to_string())

        # Update local and global data
        self.data = processed_data
        data = processed_data.copy()

        # Refresh the preprocessing table
        self._populate_table(self.data)

        # Update all tabs
        main_window = self.get_main_window()
        if main_window:
            main_window.update_tabs()
        else:
            QMessageBox.critical(self, "Error", "Could not update other tabs.")
        QMessageBox.information(self, "Success", "Preprocessing applied successfully.")

    def get_main_window(self):
        widget = self
        while widget is not None:
            if isinstance(widget, QMainWindow):
                return widget
            widget = widget.parent()
        return None