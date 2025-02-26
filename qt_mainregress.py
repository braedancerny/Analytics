import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QComboBox, QLabel, QTabWidget, QFileDialog, 
                             QMessageBox, QProgressBar, QDialog, QTreeWidget, QTreeWidgetItem, QLineEdit)
from PyQt5.QtCore import Qt
import pandas as pd
import numpy as np
from qt_single_regression import SingleRegressionTab
from qt_multiple_regression import MultipleRegressionTab
from qt_data_viewer import DataViewerTab
from qt_clustering_tab import ClusteringTab

data = None
string_mappings = {}

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Regression Analysis Tool")
        self.setGeometry(100, 100, 1200, 700)
        self.setMinimumSize(800, 400)
        self.entries = {}

        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QLabel { color: #ffffff; font-size: 14px; }
            QPushButton { background-color: #4a4a4a; color: #ffffff; border: 1px solid #555555; padding: 5px; border-radius: 3px; }
            QPushButton:hover { background-color: #666666; }
            QComboBox { background-color: #3c3c3c; color: #ffffff; border: 1px solid #555555; padding: 3px; }
            QTabWidget::pane { border: 1px solid #555555; background-color: #2b2b2b; }
            QTabBar::tab { background-color: #3c3c3c; color: #ffffff; padding: 8px; }
            QTabBar::tab:selected { background-color: #4a90e2; }
            QProgressBar { border: 1px solid #555555; background-color: #3c3c3c; color: #ffffff; }
            QProgressBar::chunk { background-color: #4a90e2; }
        """)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(10, 10, 10, 10)

        self.header = QLabel("Regression Analysis Tool")
        self.header.setStyleSheet("font-size: 18pt; font-weight: bold; color: #4a90e2;")
        self.header.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.header)

        self.button_frame = QWidget()
        self.button_layout = QHBoxLayout(self.button_frame)
        self.button_layout.setSpacing(10)
        self.load_button = QPushButton("Load Excel File")
        self.load_button.clicked.connect(self.load_data)
        self.clear_button = QPushButton("Clear All")
        self.clear_button.clicked.connect(self.clear_all)
        self.save_plot_button = QPushButton("Save Plot")
        self.save_plot_button.clicked.connect(self.save_plot)
        self.mapping_button = QPushButton("String Mappings")
        self.mapping_button.clicked.connect(self.show_string_mappings)

        self.fill_label = QLabel("Fill NaN:")
        self.fill_combo = QComboBox()
        self.fill_combo.addItems(["zero", "mean", "median", "drop"])
        self.fill_combo.setCurrentText("zero")

        for widget in [self.load_button, self.clear_button, self.save_plot_button, 
                       self.mapping_button, self.fill_label, self.fill_combo]:
            self.button_layout.addWidget(widget)
        self.button_layout.addStretch(1)
        self.layout.addWidget(self.button_frame)

        self.tabs = QTabWidget()
        self.single_tab = SingleRegressionTab(self.tabs)
        self.multiple_tab = MultipleRegressionTab(self.tabs)
        self.data_viewer_tab = DataViewerTab(self.tabs)
        self.clustering_tab = ClusteringTab(self.tabs)

        self.tabs.addTab(self.single_tab, "Single Regression")
        self.tabs.addTab(self.multiple_tab, "Multiple Regression")
        self.tabs.addTab(self.data_viewer_tab, "Data Viewer")
        self.tabs.addTab(self.clustering_tab, "Clustering")
        self.layout.addWidget(self.tabs, stretch=1)

        self.status_frame = QWidget()
        self.status_layout = QHBoxLayout(self.status_frame)
        self.progress = QProgressBar()
        self.progress.setMaximum(100)
        self.status_label = QLabel("Ready")
        self.status_layout.addWidget(self.progress, stretch=1)
        self.status_layout.addWidget(self.status_label)
        self.layout.addWidget(self.status_frame)

    def detect_header_row(self, file_path, sheet_name):
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        for i in range(min(10, df.shape[0])):
            if not df.iloc[i].isnull().all() and df.iloc[i].count() > 1:
                return i
        return 0

    def process_data(self, df, fill_method="zero"):
        global string_mappings
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')].copy()
        for col in df.columns:
            if df[col].dtype == 'object':
                # Convert to numeric, strings become NaN
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if numeric_series.isna().all() and not df[col].isna().all():
                    # Fully non-numeric column (e.g., all strings), map to integers
                    unique_values = df[col].dropna().unique()
                    if col not in string_mappings:
                        string_mappings[col] = {val: idx + 1 for idx, val in enumerate(unique_values)}
                    else:
                        current_max = max(string_mappings[col].values(), default=0)
                        for val in unique_values:
                            if val not in string_mappings[col]:
                                current_max += 1
                                string_mappings[col][val] = current_max
                    df[col] = df[col].map(string_mappings[col]).astype(float)
                else:
                    # Mixed or numeric column, keep as numeric with NaNs
                    df[col] = numeric_series
            # Ensure column is numeric (float to handle NaN)
            if df[col].dtype not in [np.float64, np.int64]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if fill_method == "zero":
            df = df.fillna(0)
        elif fill_method == "mean":
            df = df.fillna(df.mean(numeric_only=True))
        elif fill_method == "median":
            df = df.fillna(df.median(numeric_only=True))
        elif fill_method == "drop":
            df = df.dropna()
        print("Processed Data Types:", df.dtypes)
        return df

    def load_data(self):
        global data
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "Open Excel File", "", "Excel Files (*.xlsx *.xls)")
            if not file_path:
                return
            self.progress.setValue(20)
            self.status_label.setText("Loading file...")
            QApplication.processEvents()
            sheet_names = pd.ExcelFile(file_path).sheet_names
            if len(sheet_names) == 1:
                selected_sheet = sheet_names[0]
                header_row = self.detect_header_row(file_path, selected_sheet)
                data = pd.read_excel(file_path, sheet_name=selected_sheet, header=header_row)
                data = self.process_data(data, self.fill_combo.currentText())
                self.update_tabs()
            else:
                dialog = QDialog(self)
                dialog.setWindowTitle("Select Sheet and Options")
                dialog.setModal(True)
                layout = QVBoxLayout(dialog)
                layout.addWidget(QLabel("Select a sheet:"))
                sheet_combo = QComboBox()
                sheet_combo.addItems(sheet_names)
                layout.addWidget(sheet_combo)
                layout.addWidget(QLabel("Handle missing values:"))
                fill_combo = QComboBox()
                fill_combo.addItems(["zero", "mean", "median", "drop"])
                layout.addWidget(fill_combo)
                confirm_button = QPushButton("Load Sheet")
                confirm_button.clicked.connect(lambda: self.load_sheet(file_path, sheet_combo.currentText(), fill_combo.currentText(), dialog))
                layout.addWidget(confirm_button)
                dialog.exec_()
            self.progress.setValue(100)
            self.status_label.setText("Data loaded successfully.")
        except Exception as e:
            self.progress.setValue(0)
            self.status_label.setText("Error loading file.")
            QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")

    def load_sheet(self, file_path, sheet_name, fill_method, dialog):
        global data
        header_row = self.detect_header_row(file_path, sheet_name)
        try:
            data = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)
            data = self.process_data(data, fill_method)
            self.update_tabs()
            dialog.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load sheet '{sheet_name}': {str(e)}")

    def update_tabs(self):
        self.single_tab.update_dropdowns(data)
        self.multiple_tab.update_dropdowns(data)
        self.data_viewer_tab.update_table(data)
        self.clustering_tab.update_dropdowns(data)

    def clear_all(self):
        self.single_tab.clear_plot()
        self.multiple_tab.clear_results()
        self.clustering_tab.clear_plot()
        self.data_viewer_tab.update_table(data)
        self.status_label.setText("All results cleared.")

    def save_plot(self):
        if self.tabs.currentIndex() == 3:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Plot", "", "HTML Files (*.html)")
            if file_path:
                self.clustering_tab.save_plot(file_path)
                QMessageBox.information(self, "Success", "Plot saved successfully as HTML.")
        else:
            QMessageBox.information(self, "Info", "Plot saving is only available in the Clustering tab as HTML.")

    def show_string_mappings(self):
        global string_mappings, data
        if not string_mappings:
            QMessageBox.information(self, "Info", "No string mappings available.")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("String to Number Mappings")
        dialog.setGeometry(200, 200, 600, 400)
        dialog.setStyleSheet("background-color: #2b2b2b; color: #ffffff;")
        layout = QVBoxLayout(dialog)
        tree = QTreeWidget()
        tree.setHeaderLabels(["Column Name", "String Value", "Numerical Value"])
        tree.setColumnWidth(0, 150)
        tree.setColumnWidth(1, 200)
        tree.setColumnWidth(2, 100)
        tree.setStyleSheet("background-color: #3c3c3c; color: #ffffff; border: 1px solid #555555;")
        self.entries.clear()
        for col, mapping in string_mappings.items():
            for string, num in mapping.items():
                item = QTreeWidgetItem([col, str(string), str(num)])
                tree.addTopLevelItem(item)
                self.entries[(col, str(string))] = item
        tree.itemDoubleClicked.connect(self.edit_number)
        layout.addWidget(tree)
        dialog.exec_()

    def edit_number(self, item, column):
        for (col, string), tree_item in self.entries.items():
            if tree_item == item:
                current_num = item.text(2)
                edit_dialog = QDialog(self)
                edit_dialog.setWindowTitle("Edit Number")
                edit_dialog.setGeometry(250, 250, 250, 150)
                edit_dialog.setStyleSheet("background-color: #2b2b2b; color: #ffffff;")
                layout = QVBoxLayout(edit_dialog)
                layout.addWidget(QLabel(f"Column: {col}"))
                layout.addWidget(QLabel(f"String: {string}"))
                num_entry = QLineEdit(current_num)
                num_entry.setStyleSheet("background-color: #3c3c3c; color: #ffffff; border: 1px solid #555555;")
                layout.addWidget(num_entry)
                save_button = QPushButton("Save")
                save_button.clicked.connect(lambda: self.save_edit(col, string, num_entry.text(), item, edit_dialog))
                layout.addWidget(save_button)
                edit_dialog.exec_()
                break

    def save_edit(self, col, string, new_num, item, dialog):
        global string_mappings, data
        try:
            new_num = int(new_num)
            string_mappings[col][string] = new_num
            item.setText(2, str(new_num))
            if col in string_mappings:
                data[col] = data[col].map(string_mappings[col])
            self.update_tabs()
            dialog.accept()
        except ValueError:
            QMessageBox.critical(self, "Error", "Please enter a valid integer.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
