from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
                             QTreeWidget, QTreeWidgetItem)
from PyQt5.QtCore import Qt
import pandas as pd

class DataViewerTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.sort_direction = {}

        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
            }
            QLabel {
                color: #ffffff;
                font-size: 14px;
            }
            QLineEdit {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 3px;
                min-height: 25px;
            }
            QTreeWidget {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555555;
            }
        """)

        self.layout = QVBoxLayout(self)
        self.filter_frame = QWidget()
        self.filter_layout = QHBoxLayout(self.filter_frame)
        self.filter_label = QLabel("Filter:")
        self.filter_entry = QLineEdit()
        self.filter_entry.setToolTip("Type to filter rows by selected column")
        self.filter_entry.textChanged.connect(self.apply_filter)
        self.filter_layout.addWidget(self.filter_label)
        self.filter_layout.addWidget(self.filter_entry)
        self.layout.addWidget(self.filter_frame)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(False)
        self.tree.itemClicked.connect(self.on_column_click)
        self.layout.addWidget(self.tree)

    def update_table(self, data: pd.DataFrame) -> None:
        self.data = data
        self._populate_table(data)

    def _populate_table(self, data: pd.DataFrame):
        self.tree.clear()
        if data is None:
            return
        self.tree.setColumnCount(len(data.columns))
        self.tree.setHeaderLabels(list(data.columns))
        for col in range(len(data.columns)):
            self.tree.setColumnWidth(col, 100)
        for index, row in data.iterrows():
            item = QTreeWidgetItem([str(val) for val in row])
            self.tree.addTopLevelItem(item)

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
            filtered_data = self.data[self.data.apply(lambda row: any(filter_text in str(val).lower() for val in row), axis=1)]
            self._populate_table(filtered_data)

    def on_column_click(self, item, column):
        self.sort_by(column)