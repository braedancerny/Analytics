from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
                             QTreeWidget, QTreeWidgetItem, QScrollArea, QSizePolicy, QMessageBox)
from PyQt5.QtCore import Qt
import pandas as pd

class DataViewerTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.data = None
        self.sort_direction = {}
        self.is_dark_mode = True

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.filter_frame = QWidget()
        self.filter_layout = QHBoxLayout(self.filter_frame)
        self.filter_label = QLabel("Filter:")
        self.filter_label.setToolTip("Type to filter rows by selected column")
        self.update_filter_label_style()
        self.filter_entry = QLineEdit()
        self.filter_entry.textChanged.connect(self.apply_filter)
        self.filter_layout.addWidget(self.filter_label)
        self.filter_layout.addWidget(self.filter_entry, stretch=1)
        self.layout.addWidget(self.filter_frame)

        self.status_label = QLabel("Data: Raw")
        self.layout.addWidget(self.status_label)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(False)
        self.tree.itemClicked.connect(self.on_column_click)
        self.tree.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scroll_area.setWidget(self.tree)
        self.layout.addWidget(self.scroll_area, stretch=1)

    def update_filter_label_style(self):
        self.filter_label.setStyleSheet("color: #000000;")

    def update_theme(self, is_dark_mode):
        self.is_dark_mode = is_dark_mode
        self.update_filter_label_style()

    def update_table(self, data: pd.DataFrame):
        self.data = data.copy() if data is not None else None
        self.status_label.setText("Data: Preprocessed" if self.main_window.preprocessed_data is not None else "Data: Raw")
        self._populate_table(self.data)

    def _populate_table(self, data: pd.DataFrame):
        self.tree.clear()
        if data is None or data.empty:
            return
        self.tree.setColumnCount(len(data.columns))
        self.tree.setHeaderLabels(list(data.columns))
        for col in range(len(data.columns)):
            self.tree.setColumnWidth(col, 100)
        for _, row in data.iterrows():
            item = QTreeWidgetItem([str(val) if pd.notnull(val) else 'NaN' for val in row])
            self.tree.addTopLevelItem(item)
        self.tree.viewport().update()

    def sort_by(self, col):
        if self.data is None:
            return
        direction = self.sort_direction.get(col, True)
        sorted_data = self.data.sort_values(self.data.columns[col], ascending=direction)
        self.sort_direction[col] = not direction
        self._populate_table(sorted_data)

    def apply_filter(self):
        if self.data is None:
            return
        filter_text = self.filter_entry.text().lower()
        if not filter_text:
            self._populate_table(self.data)
        else:
            filtered_data = self.data[self.data.apply(lambda row: any(filter_text in str(val).lower() for val in row if pd.notnull(val)), axis=1)]
            self._populate_table(filtered_data)

    def on_column_click(self, item, column):
        self.sort_by(column)
