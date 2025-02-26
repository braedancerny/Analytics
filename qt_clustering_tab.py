from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
                             QPushButton, QCheckBox, QLineEdit, QTreeWidget, 
                             QTreeWidgetItem, QScrollArea, QMessageBox)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go

class ClusteringTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.cluster_labels = None
        self.current_algorithm = "K-Means"
        self.auto_mode = True
        self.X_scaled = None

        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
            }
            QLabel {
                color: #ffffff;
                font-size: 14px;
            }
            QPushButton {
                background-color: #4a4a4a;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 5px;
                border-radius: 3px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
            QComboBox {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 3px;
                min-height: 25px;
            }
            QLineEdit {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 3px;
                min-height: 25px;
            }
            QCheckBox {
                color: #ffffff;
            }
            QTreeWidget {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555555;
            }
        """)

        self.layout = QHBoxLayout(self)

        # Scrollable left frame
        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_frame = QWidget()
        self.left_layout = QVBoxLayout(self.left_frame)
        self.left_frame.setMinimumWidth(300)

        self.algorithm_label = QLabel("Select Clustering Algorithm:")
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["K-Means", "DBSCAN", "Hierarchical Clustering", "Gaussian Mixture Models"])
        self.algorithm_combo.setToolTip("Choose the clustering method")
        self.algorithm_combo.currentTextChanged.connect(self.update_algorithm)
        self.left_layout.addWidget(self.algorithm_label)
        self.left_layout.addWidget(self.algorithm_combo)

        self.mode_check = QCheckBox("Auto Mode")
        self.mode_check.setChecked(True)
        self.mode_check.setToolTip("Toggle between automatic and manual cluster selection")
        self.mode_check.stateChanged.connect(self.toggle_mode)
        self.left_layout.addWidget(self.mode_check)

        self.manual_frame = QWidget()
        self.manual_layout = QVBoxLayout(self.manual_frame)
        self.k_label = QLabel("Number of Clusters (k):")
        self.k_entry = QLineEdit("3")
        self.k_entry.setToolTip("Set the number of clusters (manual mode)")
        self.manual_layout.addWidget(self.k_label)
        self.manual_layout.addWidget(self.k_entry)
        self.dbscan_frame = QWidget()
        self.dbscan_layout = QVBoxLayout(self.dbscan_frame)
        self.eps_label = QLabel("DBSCAN eps:")
        self.eps_entry = QLineEdit("0.5")
        self.eps_entry.setToolTip("Set the maximum distance between points (DBSCAN)")
        self.min_samples_label = QLabel("DBSCAN min_samples:")
        self.min_samples_entry = QLineEdit("5")
        self.min_samples_entry.setToolTip("Set the minimum points per cluster (DBSCAN)")
        self.dbscan_layout.addWidget(self.eps_label)
        self.dbscan_layout.addWidget(self.eps_entry)
        self.dbscan_layout.addWidget(self.min_samples_label)
        self.dbscan_layout.addWidget(self.min_samples_entry)
        self.manual_frame.setVisible(False)
        self.left_layout.addWidget(self.manual_frame)

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

        self.elbow_button = QPushButton("Show Elbow Plot")
        self.elbow_button.setToolTip("Display the elbow plot for optimal k")
        self.elbow_button.clicked.connect(self.show_elbow_plot)
        self.left_layout.addWidget(self.elbow_button)

        self.silhouette_label = QLabel("Silhouette Score: ")
        self.left_layout.addWidget(self.silhouette_label)
        self.left_layout.addStretch()

        self.left_scroll.setWidget(self.left_frame)
        self.layout.addWidget(self.left_scroll)

        self.plot_view = QWebEngineView()
        self.plot_view.setHtml("<p>Select features and run clustering to see plot.</p>")
        self.layout.addWidget(self.plot_view, stretch=1)

    def update_dropdowns(self, data: pd.DataFrame) -> None:
        self.data = data
        numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_list.clear()
        for col in numerical_columns:
            item = QTreeWidgetItem([col])
            self.feature_list.addTopLevelItem(item)
        if len(numerical_columns) >= 2:
            self.feature_list.topLevelItem(0).setSelected(True)
            self.feature_list.topLevelItem(1).setSelected(True)

    def run_clustering(self):
        if self.data is None:
            QMessageBox.critical(self, "Error", "No data loaded.")
            return

        selected_items = self.feature_list.selectedItems()
        selected_features = [item.text(0) for item in selected_items]
        if not selected_features:
            QMessageBox.critical(self, "Error", "Please select features for clustering.")
            return

        if len(selected_features) < 2:
            QMessageBox.critical(self, "Error", "Select at least two features for clustering.")
            return

        try:
            X = self.data[selected_features].values
            if np.any(np.isnan(X)):
                QMessageBox.critical(self, "Error", "Data contains missing values. Please clean it first.")
                return
            self.X_scaled = StandardScaler().fit_transform(X)

            if self.auto_mode:
                optimal_k = self.determine_optimal_k()
            else:
                try:
                    optimal_k = int(self.k_entry.text())
                    if optimal_k < 2:
                        raise ValueError("Number of clusters must be ≥ 2")
                except ValueError as e:
                    QMessageBox.critical(self, "Error", str(e))
                    return

            model = self.get_clustering_model(optimal_k)
            self.cluster_labels = model.fit_predict(self.X_scaled)

            silhouette = silhouette_score(self.X_scaled, self.cluster_labels) if len(set(self.cluster_labels)) > 1 else 0
            self.silhouette_label.setText(f"Silhouette Score: {silhouette:.2f}")

            df = pd.DataFrame(self.X_scaled, columns=selected_features)
            df['Cluster'] = self.cluster_labels.astype(str)

            fig = px.scatter(
                df, x=selected_features[0], y=selected_features[1], color='Cluster',
                title=f"{self.current_algorithm} Clustering Results",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            self.plot_view.setHtml(fig.to_html(include_plotlyjs='cdn'))

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Clustering failed: {str(e)}")
            self.plot_view.setHtml(f"<p>Error: {str(e)}</p>")

    def get_clustering_model(self, optimal_k):
        if self.current_algorithm == "K-Means":
            return KMeans(n_clusters=optimal_k, random_state=42)
        elif self.current_algorithm == "DBSCAN":
            try:
                eps = float(self.eps_entry.text())
                min_samples = int(self.min_samples_entry.text())
            except ValueError:
                eps, min_samples = 0.5, 5
            return DBSCAN(eps=eps, min_samples=min_samples)
        elif self.current_algorithm == "Hierarchical Clustering":
            return AgglomerativeClustering(n_clusters=optimal_k)
        else:
            return GaussianMixture(n_components=optimal_k, random_state=42)

    def determine_optimal_k(self) -> int:
        k_values = range(2, min(11, len(self.X_scaled)))
        inertias = []
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.X_scaled)
            inertias.append(kmeans.inertia_)
        inertia_diffs = np.diff(inertias)
        return k_values[np.argmin(inertia_diffs) + 1]

    def show_elbow_plot(self):
        if self.X_scaled is None:
            QMessageBox.critical(self, "Error", "Run clustering first to generate data.")
            return
        k_values = range(2, min(11, len(self.X_scaled)))
        inertias = []
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.X_scaled)
            inertias.append(kmeans.inertia_)
        fig = go.Figure(data=go.Scatter(x=list(k_values), y=inertias, mode='lines+markers'))
        fig.update_layout(
            title="Elbow Plot",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Inertia"
        )
        self.plot_view.setHtml(fig.to_html(include_plotlyjs='cdn'))

    def toggle_mode(self, state):
        self.auto_mode = state == Qt.Checked
        self.manual_frame.setVisible(not self.auto_mode)
        if self.current_algorithm == "DBSCAN" and not self.auto_mode:
            self.dbscan_frame.setVisible(True)
        else:
            self.dbscan_frame.setVisible(False)

    def update_algorithm(self):
        self.current_algorithm = self.algorithm_combo.currentText()
        if not self.auto_mode and self.current_algorithm == "DBSCAN":
            self.dbscan_frame.setVisible(True)
        else:
            self.dbscan_frame.setVisible(False)
        self.manual_layout.addWidget(self.dbscan_frame)

    def clear_plot(self):
        self.plot_view.setHtml("<p>Select features and run clustering to see plot.</p>")

    def save_plot(self, file_path):
        if self.plot_view.page():
            html = self.plot_view.page().toHtml()
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html)