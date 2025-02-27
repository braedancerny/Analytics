from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QComboBox, 
                             QPushButton, QMessageBox)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt
import pandas as pd
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA

class TimeSeriesTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.data = None

        self.layout = QVBoxLayout(self)
        self.time_col_label = QLabel("Time Column:")
        self.time_col_combo = QComboBox()
        self.value_col_label = QLabel("Value Column:")
        self.value_col_combo = QComboBox()
        self.run_button = QPushButton("Run ARIMA")
        self.run_button.clicked.connect(self.run_arima)
        self.plot_view = QWebEngineView()

        self.layout.addWidget(self.time_col_label)
        self.layout.addWidget(self.time_col_combo)
        self.layout.addWidget(self.value_col_label)
        self.layout.addWidget(self.value_col_combo)
        self.layout.addWidget(self.run_button)
        self.layout.addWidget(self.plot_view)

    def update_dropdowns(self, data: pd.DataFrame):
        self.data = data
        columns = data.columns.tolist()
        self.time_col_combo.clear()
        self.value_col_combo.clear()
        self.time_col_combo.addItems(columns)
        self.value_col_combo.addItems(columns)

    def run_arima(self):
        if self.data is None:
            QMessageBox.critical(self, "Error", "No data loaded.")
            return
        time_col = self.time_col_combo.currentText()
        value_col = self.value_col_combo.currentText()
        try:
            self.data[time_col] = pd.to_datetime(self.data[time_col])
            self.data.set_index(time_col, inplace=True)
            model = ARIMA(self.data[value_col], order=(1,1,1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=10)
            fig = px.line(self.data, y=value_col, title="Time Series with Forecast")
            fig.add_scatter(x=forecast.index, y=forecast, mode='lines', name='Forecast')
            self.plot_view.setHtml(fig.to_html(include_plotlyjs='cdn'))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"ARIMA failed: {str(e)}")
