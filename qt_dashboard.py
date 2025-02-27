from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel)
from PyQt5.QtWebEngineWidgets import QWebEngineView
import pandas as pd
import plotly.express as px

class DashboardTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.data = None

        self.layout = QVBoxLayout(self)
        self.label = QLabel("Dashboard (Customize your visualizations here)")
        self.plot_view = QWebEngineView()

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.plot_view)

    def update_dashboard(self, data: pd.DataFrame):
        self.data = data
        if self.data is not None and not self.data.empty:
            fig = px.scatter(self.data, x=self.data.columns[0], y=self.data.columns[1], title="Sample Dashboard")
            self.plot_view.setHtml(fig.to_html(include_plotlyjs='cdn'))
