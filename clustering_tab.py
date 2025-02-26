import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from cefpython3 import cefpython as cef
import os

class ClusteringTab:
    def __init__(self, notebook):
        self.frame = ttk.Frame(notebook)
        self.data = None
        self.cluster_labels = None
        self.current_algorithm = "K-Means"
        self.auto_mode = True
        self.X_scaled = None
        self.fig = None
        self.browser = None

        self.main_frame = ttk.Frame(self.frame)
        self.main_frame.pack(fill="both", expand=True)

        self.left_frame = ttk.Frame(self.main_frame, width=300)
        self.left_frame.pack(side="left", fill="both", expand=False, padx=10, pady=10)

        self.graph_container = ttk.Frame(self.main_frame)
        self.graph_container.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Initialize CEF browser
        self._init_cef()

        ttk.Label(self.left_frame, text="Select Clustering Algorithm:").pack(pady=5)
        self.algorithm_dropdown = ttk.Combobox(
            self.left_frame, values=["K-Means", "DBSCAN", "Hierarchical Clustering", "Gaussian Mixture Models"],
            state="readonly"
        )
        self.algorithm_dropdown.current(0)
        self.algorithm_dropdown.pack(fill="x", pady=5)
        self.algorithm_dropdown.bind("<<ComboboxSelected>>", self.update_algorithm)
        self._add_tooltip(self.algorithm_dropdown, "Choose the clustering method")

        self.mode_var = tk.BooleanVar(value=True)
        self.mode_toggle = ttk.Checkbutton(self.left_frame, text="Auto Mode", variable=self.mode_var, command=self.toggle_mode)
        self.mode_toggle.pack(pady=5)
        self._add_tooltip(self.mode_toggle, "Toggle between automatic and manual cluster selection")

        self.manual_frame = ttk.Frame(self.left_frame)
        ttk.Label(self.manual_frame, text="Number of Clusters (k):").pack(pady=5)
        self.k_entry = ttk.Entry(self.manual_frame)
        self.k_entry.insert(0, "3")
        self.k_entry.pack(fill="x", pady=5)
        self._add_tooltip(self.k_entry, "Set the number of clusters (manual mode)")
        self.dbscan_frame = ttk.Frame(self.manual_frame)
        ttk.Label(self.dbscan_frame, text="DBSCAN eps:").pack(pady=5)
        self.eps_entry = ttk.Entry(self.dbscan_frame)
        self.eps_entry.insert(0, "0.5")
        self.eps_entry.pack(fill="x", pady=5)
        self._add_tooltip(self.eps_entry, "Set the maximum distance between points (DBSCAN)")
        ttk.Label(self.dbscan_frame, text="DBSCAN min_samples:").pack(pady=5)
        self.min_samples_entry = ttk.Entry(self.dbscan_frame)
        self.min_samples_entry.insert(0, "5")
        self.min_samples_entry.pack(fill="x", pady=5)
        self._add_tooltip(self.min_samples_entry, "Set the minimum points per cluster (DBSCAN)")
        self.manual_frame.pack_forget()

        ttk.Label(self.left_frame, text="Select Features for Clustering:").pack(pady=5)
        self.feature_listbox = tk.Listbox(self.left_frame, selectmode=tk.MULTIPLE, height=5, font=("Arial", 10))
        self.feature_listbox.pack(fill="both", expand=True, pady=5)
        self._add_tooltip(self.feature_listbox, "Select features to cluster (Ctrl+click for multiple)")

        self.run_button = ttk.Button(self.left_frame, text="Run Clustering", command=self.run_clustering)
        self.run_button.pack(fill="x", pady=5)
        self._add_tooltip(self.run_button, "Perform clustering with selected options")

        self.elbow_button = ttk.Button(self.left_frame, text="Show Elbow Plot", command=self.show_elbow_plot)
        self.elbow_button.pack(fill="x", pady=5)
        self._add_tooltip(self.elbow_button, "Display the elbow plot for optimal k")

        self.silhouette_label = ttk.Label(self.left_frame, text="Silhouette Score: ", font=("Arial", 10))
        self.silhouette_label.pack(pady=5)

    def _init_cef(self):
        cef.Initialize()
        self.browser = cef.CreateBrowserSync(
            url="about:blank",
            window_info=cef.WindowInfo(self.graph_container.winfo_id()),
            settings={"windowless_rendering_enabled": False}
        )
        self.browser.pack(fill="both", expand=True)

    def _add_tooltip(self, widget, text):
        tooltip = tk.Toplevel(widget)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_attributes("-alpha", 0.9)
        label = ttk.Label(
            tooltip, text=text, background="#333333", foreground="white",
            font=("Arial", 8), padding=2, relief="solid", borderwidth=1
        )
        label.pack()
        tooltip.withdraw()

        def show(event):
            x = widget.winfo_rootx() + 10
            y = widget.winfo_rooty() + widget.winfo_height() + 5
            tooltip.wm_geometry(f"+{x}+{y}")
            tooltip.deiconify()
        def hide(event): tooltip.withdraw()
        widget.bind("<Enter>", show)
        widget.bind("<Leave>", hide)
        widget.bind("<FocusOut>", hide)
        widget.bind("<Button-1>", hide)

    def update_dropdowns(self, data: pd.DataFrame) -> None:
        self.data = data
        numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        print(f"ClusteringTab - Numerical columns: {numerical_columns}")
        self.feature_listbox.delete(0, tk.END)
        for col in numerical_columns:
            self.feature_listbox.insert(tk.END, col)
        if len(numerical_columns) >= 2:
            self.feature_listbox.selection_set(0, 1)

    def run_clustering(self):
        if self.data is None:
            self.load_plot_in_cef("<p>No data loaded.</p>")
            return

        selected_indices = self.feature_listbox.curselection()
        if not selected_indices:
            self.load_plot_in_cef("<p>Please select features for clustering.</p>")
            return

        selected_features = [self.feature_listbox.get(i) for i in selected_indices]
        if len(selected_features) < 2:
            self.load_plot_in_cef("<p>Select at least two features for clustering.</p>")
            return

        try:
            X = self.data[selected_features].values
            if np.any(np.isnan(X)):
                self.load_plot_in_cef("<p>Data contains missing values. Please clean it first.</p>")
                return
            self.X_scaled = StandardScaler().fit_transform(X)

            if self.auto_mode:
                optimal_k = self.determine_optimal_k()
            else:
                try:
                    optimal_k = int(self.k_entry.get())
                    if optimal_k < 2:
                        raise ValueError("Number of clusters must be ≥ 2")
                except ValueError as e:
                    self.load_plot_in_cef(f"<p>Error: {str(e)}</p>")
                    return

            model = self.get_clustering_model(optimal_k)
            self.cluster_labels = model.fit_predict(self.X_scaled)

            silhouette = silhouette_score(self.X_scaled, self.cluster_labels) if len(set(self.cluster_labels)) > 1 else 0
            self.silhouette_label.config(text=f"Silhouette Score: {silhouette:.2f}")

            df = pd.DataFrame(self.X_scaled, columns=selected_features)
            df['Cluster'] = self.cluster_labels.astype(str)
            print(f"Clustering data: {df.head()}")
            self.fig = px.scatter(
                df, x=selected_features[0], y=selected_features[1], color='Cluster',
                title=f"{self.current_algorithm} Clustering Results",
                hover_data=selected_features,
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            html_content = self.fig.to_html(include_plotlyjs=True)
            self.load_plot_in_cef(html_content)

        except Exception as e:
            self.load_plot_in_cef(f"<p>Error: {str(e)}</p>")
            messagebox.showerror("Error", f"Clustering failed: {str(e)}")

    def get_clustering_model(self, optimal_k):
        if self.current_algorithm == "K-Means":
            return KMeans(n_clusters=optimal_k, random_state=42)
        elif self.current_algorithm == "DBSCAN":
            try:
                eps = float(self.eps_entry.get())
                min_samples = int(self.min_samples_entry.get())
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
            self.load_plot_in_cef("<p>Run clustering first to generate data.</p>")
            return
        k_values = range(2, min(11, len(self.X_scaled)))
        inertias = []
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.X_scaled)
            inertias.append(kmeans.inertia_)
        print(f"Elbow data: k_values={k_values}, inertias={inertias[:5]}")
        self.fig = go.Figure(data=go.Scatter(x=list(k_values), y=inertias, mode='lines+markers'))
        self.fig.update_layout(
            title="Elbow Plot",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Inertia"
        )
        html_content = self.fig.to_html(include_plotlyjs=True)
        self.load_plot_in_cef(html_content)

    def toggle_mode(self):
        self.auto_mode = self.mode_var.get()
        if self.auto_mode:
            self.manual_frame.pack_forget()
        else:
            self.manual_frame.pack(after=self.mode_toggle)
            if self.current_algorithm == "DBSCAN":
                self.dbscan_frame.pack()
            else:
                self.dbscan_frame.pack_forget()

    def update_algorithm(self, event=None):
        self.current_algorithm = self.algorithm_dropdown.get()
        if not self.auto_mode and self.current_algorithm == "DBSCAN":
            self.dbscan_frame.pack()
        elif not self.auto_mode:
            self.dbscan_frame.pack_forget()

    def clear_plot(self):
        if self.browser:
            self.browser.LoadUrl("about:blank")

    def save_plot(self, file_path):
        if self.fig is not None:
            self.fig.write_html(file_path, include_plotlyjs=True)

    def load_plot_in_cef(self, html_content):
        temp_file = "temp_clustering.html"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        self.browser.LoadUrl(f"file://{os.path.abspath(temp_file)}")