from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
                             QPushButton, QTreeWidget, QTreeWidgetItem, QMessageBox, 
                             QScrollArea, QSizePolicy)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

class ClusteringTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.data = None
        self.is_dark_mode = True

        self.setStyleSheet("QLabel { color: #000000; }")

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_frame = QWidget()
        self.left_layout = QVBoxLayout(self.left_frame)
        self.left_layout.setAlignment(Qt.AlignTop)
        self.left_frame.setMinimumWidth(250)

        self.algorithm_label = QLabel("Select Clustering Algorithm:")
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["K-Means"])
        self.algorithm_combo.setToolTip("Choose the clustering method")
        self.left_layout.addWidget(self.algorithm_label)
        self.left_layout.addWidget(self.algorithm_combo)

        self.feature_label = QLabel("Select Features for Clustering:")
        self.feature_list = QTreeWidget()
        self.feature_list.setHeaderHidden(True)
        self.feature_list.setSelectionMode(QTreeWidget.MultiSelection)
        self.feature_list.setToolTip("Select features to cluster (Ctrl+click for multiple)")
        self.left_layout.addWidget(self.feature_label)
        self.left_layout.addWidget(self.feature_list)

        self.run_button = QPushButton("Run Clustering")
        self.run_button.setToolTip("Perform clustering with selected options")
        self.run_button.clicked.connect(self.run_clustering)
        self.left_layout.addWidget(self.run_button)

        self.left_layout.addStretch(1)
        self.left_scroll.setWidget(self.left_frame)
        self.layout.addWidget(self.left_scroll, stretch=1)

        self.plot_view = QWebEngineView()
        self.plot_view.setHtml("<p>Select features and run clustering to see plot.</p>")
        self.plot_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout.addWidget(self.plot_view, stretch=3)

    def update_theme(self, is_dark_mode):
        self.is_dark_mode = is_dark_mode

    def update_dropdowns(self, data: pd.DataFrame):
        self.data = data.copy() if data is not None else None
        all_columns = data.columns.tolist() if data is not None else []
        self.feature_list.clear()
        for col in all_columns:
            item = QTreeWidgetItem([col])
            self.feature_list.addTopLevelItem(item)
        if len(all_columns) >= 2:
            self.feature_list.topLevelItem(0).setSelected(True)
            self.feature_list.topLevelItem(1).setSelected(True)

    def run_clustering(self):
        if self.main_window.preprocessed_data is None:
            QMessageBox.warning(self, "Warning", "Please preprocess the data first.")
            return
        if self.data is None:
            QMessageBox.critical(self, "Error", "No data loaded.")
            return

        selected_items = self.feature_list.selectedItems()
        selected_features = [item.text(0) for item in selected_items]
        if len(selected_features) < 2:
            QMessageBox.critical(self, "Error", "Select at least two features for clustering.")
            return

        try:
            X = self.data[selected_features].values
            if np.any(np.isnan(X)):
                QMessageBox.critical(self, "Error", "Data contains missing values after preprocessing.")
                return
            X_scaled = StandardScaler().fit_transform(X)

            kmeans = KMeans(n_clusters=3, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)

            df = pd.DataFrame(X_scaled, columns=selected_features)
            df['Cluster'] = cluster_labels.astype(str)

            fig = px.scatter(df, x=selected_features[0], y=selected_features[1], color='Cluster',
                             title="K-Means Clustering Results")
            self.plot_view.setHtml(fig.to_html(include_plotlyjs='cdn'))

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Clustering failed: {str(e)}")
            self.plot_view.setHtml(f"<p>Error: {str(e)}</p>")

    def clear_plot(self):
        self.plot_view.setHtml("<p>Select features and run clustering to see plot.</p>")
