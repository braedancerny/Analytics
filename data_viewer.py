import tkinter as tk
from tkinter import ttk
import pandas as pd

class DataViewerTab:
    def __init__(self, notebook):
        self.frame = ttk.Frame(notebook)
        self.data = None

        self.main_frame = ttk.Frame(self.frame)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.filter_frame = ttk.Frame(self.main_frame)
        self.filter_frame.pack(fill="x", pady=5)
        ttk.Label(self.filter_frame, text="Filter:").pack(side="left", padx=5)
        self.filter_entry = ttk.Entry(self.filter_frame)
        self.filter_entry.pack(side="left", fill="x", expand=True, padx=5)
        self.filter_entry.bind("<KeyRelease>", self.apply_filter)
        self._add_tooltip(self.filter_entry, "Type to filter rows by selected column")

        self.tree = ttk.Treeview(self.main_frame, show="headings")
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.bind("<Button-1>", self.on_column_click)

        self.scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.tree.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=self.scrollbar.set)

        self.sort_direction = {}

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

    def update_table(self, data: pd.DataFrame) -> None:
        self.data = data
        self._populate_table(data)

    def _populate_table(self, data: pd.DataFrame):
        for row in self.tree.get_children():
            self.tree.delete(row)
        self.tree["columns"] = list(data.columns)
        for col in data.columns:
            self.tree.heading(col, text=col, command=lambda c=col: self.sort_by(c))
            self.tree.column(col, width=100, anchor="center", stretch=tk.YES)
        for index, row in data.iterrows():
            self.tree.insert("", "end", values=list(row))

    def sort_by(self, col):
        if self.data is None:
            return
        direction = self.sort_direction.get(col, True)
        sorted_data = self.data.sort_values(col, ascending=direction)
        self.sort_direction[col] = not direction
        self._populate_table(sorted_data)

    def apply_filter(self, event):
        if self.data is None:
            return
        filter_text = self.filter_entry.get().lower()
        if not filter_text:
            self._populate_table(self.data)
        else:
            filtered_data = self.data[self.data.apply(lambda row: any(filter_text in str(val).lower() for val in row), axis=1)]
            self._populate_table(filtered_data)

    def on_column_click(self, event):
        region = self.tree.identify("region", event.x, event.y)
        if region == "heading":
            col = self.tree.identify_column(event.x)[1:]
            col_name = self.tree["columns"][int(col) - 1]
            self.sort_by(col_name)