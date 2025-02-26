import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QComboBox, QLabel, QTabWidget, QFileDialog, 
                             QMessageBox, QProgressBar, QDialog, QTreeWidget, QTreeWidgetItem)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import pandas as pd
import numpy as np
from qt_single_regression import SingleRegressionTab
from qt_multiple_regression import MultipleRegressionTab
from qt_data_viewer import DataViewerTab
from qt_clustering_tab import ClusteringTab
from qt_correlation_heatmap import CorrelationHeatmapTab
from qt_data_preprocessing import DataPreprocessingTab

data = None
string_mappings = {}

class LoadDataThread(QThread):
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)

    def __init__(self, file_path, sheet_name=None, header_row=0):
        super().__init__()
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.header_row = header_row

    def run(self):
        try:
            if self.file_path.endswith('.xlsx') or self.file_path.endswith('.xls'):
                if self.sheet_name:
                    df = pd.read_excel(self.file_path, sheet_name=self.sheet_name, header=self.header_row)
                else:
                    df = pd.read_excel(self.file_path, header=self.header_row)
            else:
                df = pd.read_csv(self.file_path)
            self.finished.emit(df)
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Regression Analysis Tool")
        self.setGeometry(100, 100, 1200, 700)
        self.setMinimumSize(800, 400)
        self.entries = {}

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(10, 10, 10, 10)

        self.header = QLabel("Regression Analysis Tool")
        self.header.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.header)

        self.button_frame = QWidget()
        self.button_layout = QHBoxLayout(self.button_frame)
        self.load_button = QPushButton("Load Excel File")
        self.load_button.setToolTip("Load an Excel or CSV file")
        self.load_button.clicked.connect(self.load_data)
        self.clear_button = QPushButton("Clear All")
        self.clear_button.setToolTip("Reset all tabs and data")
        self.clear_button.clicked.connect(self.clear_all)
        self.save_plot_button = QPushButton("Export Plot")
        self.save_plot_button.setToolTip("Export the current tab's plot as HTML")
        self.save_plot_button.clicked.connect(self.export_plot)
        self.save_results_button = QPushButton("Export Results")
        self.save_results_button.setToolTip("Export regression results as CSV")
        self.save_results_button.clicked.connect(self.export_results)
        self.theme_button = QPushButton("Toggle Theme")
        self.theme_button.setToolTip("Switch between dark and light themes")
        self.theme_button.clicked.connect(self.toggle_theme)

        for widget in [self.load_button, self.clear_button, self.save_plot_button, 
                       self.save_results_button, self.theme_button]:
            self.button_layout.addWidget(widget)
        self.button_layout.addStretch(1)
        self.layout.addWidget(self.button_frame)

        self.tabs = QTabWidget()
        self.single_tab = SingleRegressionTab(self.tabs)
        self.multiple_tab = MultipleRegressionTab(self.tabs)
        self.data_viewer_tab = DataViewerTab(self.tabs)
        self.clustering_tab = ClusteringTab(self.tabs)
        self.correlation_tab = CorrelationHeatmapTab(self.tabs)
        self.preprocessing_tab = DataPreprocessingTab(self.tabs)

        self.tabs.addTab(self.preprocessing_tab, "Data Preprocessing")
        self.tabs.addTab(self.correlation_tab, "Correlation Heatmap")
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

        self.dark_stylesheet = """
            QMainWindow { 
                background-color: #2b2b2b; 
            }
            QLabel { 
                color: #e0e0e0; 
                font-family: Segoe UI, Arial, sans-serif; 
                font-size: 14pt; 
            }
            QPushButton { 
                background-color: #4a90e2; 
                color: #ffffff; 
                border: none; 
                border-radius: 5px; 
                padding: 5px 10px; 
                font-family: Segoe UI, Arial, sans-serif; 
                font-size: 12pt; 
            }
            QPushButton:hover { 
                background-color: #357abd; 
            }
            QComboBox { 
                background-color: #3c3c3c; 
                color: #e0e0e0; 
                border: 1px solid #555555; 
                border-radius: 3px; 
                padding: 3px; 
                font-family: Segoe UI, Arial, sans-serif; 
                font-size: 12pt; 
            }
            QComboBox::drop-down { 
                border: none; 
            }
            QComboBox QAbstractItemView { 
                background-color: #3c3c3c; 
                color: #e0e0e0; 
                selection-background-color: #4a90e2; 
            }
            QLineEdit { 
                background-color: #3c3c3c; 
                color: #e0e0e0; 
                border: 1px solid #555555; 
                border-radius: 3px; 
                padding: 3px; 
                font-family: Segoe UI, Arial, sans-serif; 
                font-size: 12pt; 
            }
            QTabWidget::pane { 
                border: 1px solid #555555; 
                background-color: #2b2b2b; 
            }
            QTabBar::tab { 
                background-color: #3c3c3c; 
                color: #e0e0e0; 
                padding: 8px; 
                border-top-left-radius: 5px; 
                border-top-right-radius: 5px; 
            }
            QTabBar::tab:selected { 
                background-color: #4a90e2; 
                color: #ffffff; 
            }
            QTreeWidget { 
                background-color: #3c3c3c; 
                color: #e0e0e0; 
                border: 1px solid #555555; 
                alternate-background-color: #353535; 
            }
            QTreeWidget::item:hover { 
                background-color: #4a4a4a; 
            }
            QProgressBar { 
                border: 1px solid #555555; 
                background-color: #3c3c3c; 
                color: #e0e0e0; 
                text-align: center; 
            }
            QProgressBar::chunk { 
                background-color: #4a90e2; 
            }
            QScrollArea { 
                background-color: #2b2b2b; 
                border: none; 
            }
            QToolTip { 
                background-color: #3c3c3c; 
                color: #e0e0e0; 
                border: 1px solid #555555; 
                font-family: Segoe UI, Arial, sans-serif; 
                font-size: 10pt; 
            }
        """
        self.light_stylesheet = """
            QMainWindow { 
                background-color: #f5f5f5; 
            }
            QLabel { 
                color: #1A1A1A; 
                font-family: Segoe UI, Arial, sans-serif; 
                font-size: 14pt; 
            }
            QPushButton { 
                background-color: #4a90e2; 
                color: #ffffff; 
                border: none; 
                border-radius: 5px; 
                padding: 5px 10px; 
                font-family: Segoe UI, Arial, sans-serif; 
                font-size: 12pt; 
            }
            QPushButton:hover { 
                background-color: #357abd; 
            }
            QComboBox { 
                background-color: #ffffff; 
                color: #1A1A1A; 
                border: 1px solid #cccccc; 
                border-radius: 3px; 
                padding: 3px; 
                font-family: Segoe UI, Arial, sans-serif; 
                font-size: 12pt; 
            }
            QComboBox::drop-down { 
                border: none; 
            }
            QComboBox QAbstractItemView { 
                background-color: #ffffff; 
                color: #1A1A1A; 
                selection-background-color: #4a90e2; 
            }
            QLineEdit { 
                background-color: #ffffff; 
                color: #1A1A1A; 
                border: 1px solid #cccccc; 
                border-radius: 3px; 
                padding: 3px; 
                font-family: Segoe UI, Arial, sans-serif; 
                font-size: 12pt; 
            }
            QTabWidget::pane { 
                border: 1px solid #cccccc; 
                background-color: #f5f5f5; 
            }
            QTabBar::tab { 
                background-color: #e0e0e0; 
                color: #1A1A1A; 
                padding: 8px; 
                border-top-left-radius: 5px; 
                border-top-right-radius: 5px; 
            }
            QTabBar::tab:selected { 
                background-color: #4a90e2; 
                color: #ffffff; 
            }
            QTreeWidget { 
                background-color: #ffffff; 
                color: #1A1A1A; 
                border: 1px solid #cccccc; 
                alternate-background-color: #fafafa; 
            }
            QTreeWidget::item:hover { 
                background-color: #e5e5e5; 
            }
            QProgressBar { 
                border: 1px solid #cccccc; 
                background-color: #ffffff; 
                color: #1A1A1A; 
                text-align: center; 
            }
            QProgressBar::chunk { 
                background-color: #4a90e2; 
            }
            QScrollArea { 
                background-color: #f5f5f5; 
                border: none; 
            }
            QToolTip { 
                background-color: #ffffff; 
                color: #1A1A1A; 
                border: 1px solid #cccccc; 
                font-family: Segoe UI, Arial, sans-serif; 
                font-size: 10pt; 
            }
        """
        self.is_dark_mode = True
        self.setStyleSheet(self.dark_stylesheet)
        self.header.setStyleSheet("font-size: 18pt; font-weight: bold; color: #4a90e2;")

    def detect_header_row(self, file_path, sheet_name):
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        for i in range(min(10, df.shape[0])):
            if not df.iloc[i].isnull().all() and df.iloc[i].count() > 1:
                return i
        return 0

    def process_data(self, df):
        global string_mappings
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')].copy()

        for col in df.columns:
            # Check if the column is mostly numeric (e.g., >50% numeric values)
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            numeric_ratio = numeric_series.notna().mean()
            
            if numeric_ratio > 0.5:  # Mostly numeric column
                # Convert non-numeric values to NaN, keep numeric values as is
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                # Handle categorical columns (mostly non-numeric)
                if df[col].dtype == 'object':
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

        # Return DataFrame with NaN values intact
        return df

    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Excel Files (*.xlsx *.xls);;CSV Files (*.csv)")
        if not file_path:
            return
        self.progress.setValue(20)
        self.status_label.setText("Loading file...")
        QApplication.processEvents()
        
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            sheet_names = pd.ExcelFile(file_path).sheet_names
            if len(sheet_names) == 1:
                header_row = self.detect_header_row(file_path, sheet_names[0])
                self.thread = LoadDataThread(file_path, sheet_names[0], header_row)
                self.thread.finished.connect(self.on_data_loaded)
                self.thread.error.connect(self.on_load_error)
                self.thread.start()
            else:
                dialog = QDialog(self)
                dialog.setWindowTitle("Select Sheet and Options")
                dialog.setModal(True)
                layout = QVBoxLayout(dialog)
                layout.addWidget(QLabel("Select a sheet:"))
                sheet_combo = QComboBox()
                sheet_combo.addItems(sheet_names)
                layout.addWidget(sheet_combo)
                confirm_button = QPushButton("Load Sheet")
                confirm_button.clicked.connect(lambda: self.load_sheet(file_path, sheet_combo.currentText(), dialog))
                layout.addWidget(confirm_button)
                dialog.exec_()
        else:
            self.thread = LoadDataThread(file_path)
            self.thread.finished.connect(self.on_data_loaded)
            self.thread.error.connect(self.on_load_error)
            self.thread.start()

    def load_sheet(self, file_path, sheet_name, dialog):
        header_row = self.detect_header_row(file_path, sheet_name)
        self.thread = LoadDataThread(file_path, sheet_name, header_row)
        self.thread.finished.connect(self.on_data_loaded)
        self.thread.error.connect(self.on_load_error)
        self.thread.start()
        dialog.accept()

    def on_data_loaded(self, df):
        global data
        data = self.process_data(df)
        self.update_tabs()
        self.progress.setValue(100)
        self.status_label.setText("Data loaded successfully.")

    def on_load_error(self, error_msg):
        self.progress.setValue(0)
        self.status_label.setText("Error loading file.")
        QMessageBox.critical(self, "Error", f"Failed to load file: {error_msg}")

    def update_tabs(self):
        global data
        self.single_tab.update_dropdowns(data)
        self.multiple_tab.update_dropdowns(data)
        self.data_viewer_tab.update_table(data)
        self.clustering_tab.update_dropdowns(data)
        self.correlation_tab.update_dropdowns(data)
        self.preprocessing_tab.update_dropdowns(data)

    def clear_all(self):
        global data
        data = None
        self.single_tab.clear_plot()
        self.multiple_tab.clear_results()
        self.clustering_tab.clear_plot()
        self.data_viewer_tab.update_table(None)
        self.correlation_tab.plot_view.setHtml("<p>No data loaded.</p>")
        self.preprocessing_tab.update_dropdowns(None)
        self.status_label.setText("All results cleared.")

    def export_plot(self):
        current_tab = self.tabs.currentWidget()
        if hasattr(current_tab, 'plot_view'):
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Plot", "", "HTML Files (*.html)")
            if file_path:
                current_tab.plot_view.page().toHtml(lambda html: self.save_html(file_path, html))
                QMessageBox.information(self, "Success", "Plot saved successfully as HTML.")

    def save_html(self, file_path, html):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html)

    def export_results(self):
        current_tab = self.tabs.currentWidget()
        if hasattr(current_tab, 'results_layout'):
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "CSV Files (*.csv)")
            if file_path:
                results = {}
                for i in range(current_tab.results_layout.count()):
                    label = current_tab.results_layout.itemAt(i).widget()
                    if isinstance(label, QLabel):
                        key, value = label.text().split(": ")
                        try:
                            results[key] = float(value)
                        except ValueError:
                            results[key] = value
                pd.DataFrame([results]).to_csv(file_path, index=False)
                QMessageBox.information(self, "Success", "Results saved successfully.")

    def toggle_theme(self):
        if self.is_dark_mode:
            self.setStyleSheet(self.light_stylesheet)
            self.header.setStyleSheet("font-size: 18pt; font-weight: bold; color: #4a90e2;")
            self.single_tab.update_theme(False)
            self.multiple_tab.update_theme(False)
            self.data_viewer_tab.update_theme(False)
            self.clustering_tab.update_theme(False)
            self.correlation_tab.update_theme(False)
            self.preprocessing_tab.update_theme(False)
        else:
            self.setStyleSheet(self.dark_stylesheet)
            self.header.setStyleSheet("font-size: 18pt; font-weight: bold; color: #4a90e2;")
            self.single_tab.update_theme(True)
            self.multiple_tab.update_theme(True)
            self.data_viewer_tab.update_theme(True)
            self.clustering_tab.update_theme(True)
            self.correlation_tab.update_theme(True)
            self.preprocessing_tab.update_theme(True)
        self.is_dark_mode = not self.is_dark_mode

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
