from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QComboBox, 
                             QPushButton, QMessageBox, QTableWidget, QTableWidgetItem)
from PyQt5.QtCore import QThread, pyqtSignal
import pandas as pd

class PreprocessThread(QThread):
    finished = pyqtSignal(pd.DataFrame)

    def __init__(self, data, missing_method, scaling_method):
        super().__init__()
        self.data = data
        self.missing_method = missing_method
        self.scaling_method = scaling_method

    def run(self):
        processed_data = self.data.copy()
        if self.missing_method == "Drop":
            processed_data.dropna(inplace=True)
        elif self.missing_method == "Fill with Mean":
            processed_data.fillna(processed_data.mean(), inplace=True)
        if self.scaling_method == "Standardize":
            processed_data = (processed_data - processed_data.mean()) / processed_data.std()
        elif self.scaling_method == "Normalize":
            processed_data = (processed_data - processed_data.min()) / (processed_data.max() - processed_data.min())
        self.finished.emit(processed_data)

class DataPreprocessingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.data = None

        self.layout = QVBoxLayout(self)
        self.missing_label = QLabel("Handle Missing Values:")
        self.missing_combo = QComboBox()
        self.missing_combo.addItems(["Drop", "Fill with Mean"])
        self.scaling_label = QLabel("Scaling Method:")
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems(["None", "Standardize", "Normalize"])
        self.apply_button = QPushButton("Apply Preprocessing")
        self.apply_button.clicked.connect(self.apply_preprocessing)
        self.table = QTableWidget()

        self.layout.addWidget(self.missing_label)
        self.layout.addWidget(self.missing_combo)
        self.layout.addWidget(self.scaling_label)
        self.layout.addWidget(self.scaling_combo)
        self.layout.addWidget(self.apply_button)
        self.layout.addWidget(self.table)

    def update_data(self, data: pd.DataFrame):
        self.data = data
        self._populate_table(self.data)

    def _populate_table(self, data):
        self.table.clear()
        self.table.setRowCount(data.shape[0])
        self.table.setColumnCount(data.shape[1])
        self.table.setHorizontalHeaderLabels(data.columns)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                self.table.setItem(i, j, QTableWidgetItem(str(data.iloc[i, j])))

    def apply_preprocessing(self):
        if self.main_window.original_data is None:
            QMessageBox.critical(self, "Error", "No data loaded.")
            return
        self.thread = PreprocessThread(self.main_window.original_data, self.missing_combo.currentText(), self.scaling_combo.currentText())
        self.thread.finished.connect(self.on_preprocessing_finished)
        self.thread.start()

    def on_preprocessing_finished(self, processed_data):
        self.data = processed_data
        self.main_window.preprocessed_data = processed_data
        self._populate_table(self.data)
        self.main_window.save_state()
        self.main_window.update_tabs()
        QMessageBox.information(self, "Success", "Preprocessing applied successfully.")
