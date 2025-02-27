from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QComboBox, 
                             QPushButton, QMessageBox, QTreeWidget, QTreeWidgetItem)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import plotly.express as px

class MachineLearningTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.data = None

        self.layout = QVBoxLayout(self)
        self.target_label = QLabel("Target Variable:")
        self.target_combo = QComboBox()
        self.feature_label = QLabel("Features:")
        self.feature_list = QTreeWidget()
        self.run_button = QPushButton("Run Decision Tree")
        self.run_button.clicked.connect(self.run_decision_tree)
        self.plot_view = QWebEngineView()

        self.layout.addWidget(self.target_label)
        self.layout.addWidget(self.target_combo)
        self.layout.addWidget(self.feature_label)
        self.layout.addWidget(self.feature_list)
        self.layout.addWidget(self.run_button)
        self.layout.addWidget(self.plot_view)

    def update_dropdowns(self, data: pd.DataFrame):
        self.data = data
        columns = data.columns.tolist()
        self.target_combo.clear()
        self.feature_list.clear()
        self.target_combo.addItems(columns)
        for col in columns:
            item = QTreeWidgetItem([col])
            self.feature_list.addTopLevelItem(item)

    def run_decision_tree(self):
        if self.data is None:
            QMessageBox.critical(self, "Error", "No data loaded.")
            return
        target_col = self.target_combo.currentText()
        selected_items = self.feature_list.selectedItems()
        features = [item.text(0) for item in selected_items]
        if not features:
            QMessageBox.critical(self, "Error", "Select at least one feature.")
            return
        try:
            X = self.data[features]
            y = self.data[target_col]
            model = DecisionTreeClassifier()
            model.fit(X, y)
            y_pred = model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            fig = px.scatter(self.data, x=features[0], y=target_col, title=f"Decision Tree Accuracy: {accuracy:.2f}")
            self.plot_view.setHtml(fig.to_html(include_plotlyjs='cdn'))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Decision Tree failed: {str(e)}")
