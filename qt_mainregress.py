import sys
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
                             QPushButton, QFileDialog, QInputDialog, QMessageBox, QHBoxLayout)
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWebEngineWidgets import QWebEngineView
import sqlite3
from cryptography.fernet import Fernet
from qt_time_series import TimeSeriesTab
from qt_machine_learning import MachineLearningTab
from qt_statistical_tests import StatisticalTestsTab
from qt_dashboard import DashboardTab
from qt_preprocessing import DataPreprocessingTab
from qt_single_regression import SingleRegressionTab
from qt_multiple_regression import MultipleRegressionTab
from qt_data_viewer import DataViewerTab
from qt_clustering_tab import ClusteringTab
from qt_correlation_heatmap import CorrelationHeatmapTab

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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analytics Program")
        self.setGeometry(100, 100, 1200, 800)

        # Data attributes
        self.original_data = None
        self.preprocessed_data = None
        self.data_history = []
        self.history_index = -1

        # Encryption key
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)

        # UI setup
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Tab widget
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        # Initialize tabs
        self.preprocessing_tab = DataPreprocessingTab(self)
        self.time_series_tab = TimeSeriesTab(self)
        self.ml_tab = MachineLearningTab(self)
        self.stats_tab = StatisticalTestsTab(self)
        self.dashboard_tab = DashboardTab(self)
        self.single_regression_tab = SingleRegressionTab(self)
        self.multiple_regression_tab = MultipleRegressionTab(self)
        self.data_viewer_tab = DataViewerTab(self)
        self.clustering_tab = ClusteringTab(self)
        self.correlation_tab = CorrelationHeatmapTab(self)

        self.tabs.addTab(self.preprocessing_tab, "Data Preprocessing")
        self.tabs.addTab(self.time_series_tab, "Time Series Analysis")
        self.tabs.addTab(self.ml_tab, "Machine Learning")
        self.tabs.addTab(self.stats_tab, "Statistical Tests")
        self.tabs.addTab(self.dashboard_tab, "Dashboard")
        self.tabs.addTab(self.single_regression_tab, "Single Regression")
        self.tabs.addTab(self.multiple_regression_tab, "Multiple Regression")
        self.tabs.addTab(self.data_viewer_tab, "Data Viewer")
        self.tabs.addTab(self.clustering_tab, "Clustering")
        self.tabs.addTab(self.correlation_tab, "Correlation Heatmap")

        # Button layout
        self.button_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.load_data)
        self.report_button = QPushButton("Generate Report")
        self.report_button.clicked.connect(self.generate_report)
        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.undo)
        self.redo_button = QPushButton("Redo")
        self.redo_button.clicked.connect(self.redo)
        
        self.button_layout.addWidget(self.load_button)
        self.button_layout.addWidget(self.report_button)
        self.button_layout.addWidget(self.undo_button)
        self.button_layout.addWidget(self.redo_button)
        self.layout.addLayout(self.button_layout)

        # Stylesheet
        self.dark_stylesheet = """
            QWidget {
                background-color: #2b2b2b;
                color: #e0e0e0;
            }
            QTabWidget::pane {
                border: 1px solid #555555;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #3c3c3c;
                color: #e0e0e0;
                padding: 10px;
            }
            QTabBar::tab:selected {
                background-color: #4a90e2;
                color: #ffffff;
            }
            QPushButton {
                background-color: #4a90e2;
                color: #ffffff;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QComboBox, QLineEdit, QTreeWidget {
                background-color: #3c3c3c;
                color: #e0e0e0;
                border: 1px solid #555555;
            }
        """
        self.setStyleSheet(self.dark_stylesheet)

    def load_sql_data(self):
        try:
            conn = sqlite3.connect('database.db')
            query = "SELECT * FROM table_name"
            df = pd.read_sql_query(query, conn)
            self.on_data_loaded(df)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"SQL Load Failed: {str(e)}")

    def load_data(self):
        options = ["Excel/CSV", "SQL"]
        choice, _ = QInputDialog.getItem(self, "Select Data Source", "Choose data source:", options, 0, False)
        if choice == "Excel/CSV":
            file_name, _ = QFileDialog.getOpenFileName(self, "Open Data File", "", "CSV Files (*.csv);;Excel Files (*.xlsx)")
            if file_name:
                try:
                    if file_name.endswith('.csv'):
                        df = pd.read_csv(file_name)
                    else:
                        df = pd.read_excel(file_name)
                    self.on_data_loaded(df)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")
        elif choice == "SQL":
            self.load_sql_data()

    def on_data_loaded(self, df):
        self.original_data = df
        self.preprocessed_data = df.copy()
        self.save_state()
        self.update_tabs()
        QMessageBox.information(self, "Success", "Data loaded successfully.")

    def update_tabs(self):
        self.preprocessing_tab.update_data(self.preprocessed_data)
        self.time_series_tab.update_dropdowns(self.preprocessed_data)
        self.ml_tab.update_dropdowns(self.preprocessed_data)
        self.stats_tab.update_dropdowns(self.preprocessed_data)
        self.dashboard_tab.update_dashboard(self.preprocessed_data)
        self.single_regression_tab.update_dropdowns(self.preprocessed_data)
        self.multiple_regression_tab.update_dropdowns(self.preprocessed_data)
        self.data_viewer_tab.update_table(self.preprocessed_data)
        self.clustering_tab.update_dropdowns(self.preprocessed_data)
        self.correlation_tab.update_dropdowns(self.preprocessed_data)

    def save_state(self):
        self.data_history = self.data_history[:self.history_index + 1]
        self.data_history.append(self.preprocessed_data.copy())
        self.history_index += 1

    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.preprocessed_data = self.data_history[self.history_index]
            self.update_tabs()

    def redo(self):
        if self.history_index < len(self.data_history) - 1:
            self.history_index += 1
            self.preprocessed_data = self.data_history[self.history_index]
            self.update_tabs()

    def generate_report(self):
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        try:
            c = canvas.Canvas("report.pdf", pagesize=letter)
            c.drawString(100, 750, "Analytics Report")
            c.drawString(100, 730, f"Data Shape: {self.preprocessed_data.shape}")
            c.save()
            QMessageBox.information(self, "Success", "Report generated as report.pdf")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Report generation failed: {str(e)}")

    def encrypt_data(self, df):
        df_str = df.to_csv()
        encrypted = self.cipher.encrypt(df_str.encode())
        with open("encrypted_data.dat", "wb") as f:
            f.write(encrypted)

    def decrypt_data(self):
        with open("encrypted_data.dat", "rb") as f:
            encrypted = f.read()
        decrypted = self.cipher.decrypt(encrypted).decode()
        df = pd.read_csv(pd.compat.StringIO(decrypted))
        return df

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
