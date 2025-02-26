import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import pandas as pd
import numpy as np
import sys
from single_regression import SingleRegressionTab
from multiple_regression import MultipleRegressionTab
from data_viewer import DataViewerTab
from clustering_tab import ClusteringTab

data = None
string_mappings = {}

def detect_header_row(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    for i in range(min(10, df.shape[0])):
        if not df.iloc[i].isnull().all() and df.iloc[i].count() > 1:
            return i
    return 0

def process_data(df, fill_method="zero"):
    global string_mappings
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')].copy()
    print("Initial dtypes:", df.dtypes)
    for col in df.columns:
        try:
            df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
        except ValueError:
            pass
        if df[col].dtype == 'object':
            test_series = pd.to_numeric(df[col], errors='coerce')
            if test_series.isna().all():
                unique_values = df[col].dropna().unique()
                if col not in string_mappings:
                    string_mappings[col] = {val: idx for idx, val in enumerate(unique_values)}
                else:
                    current_max = max(string_mappings[col].values(), default=-1)
                    for val in unique_values:
                        if val not in string_mappings[col]:
                            current_max += 1
                            string_mappings[col][val] = current_max
                df.loc[:, col] = df[col].map(string_mappings[col])
            else:
                df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
    if fill_method == "zero":
        df = df.fillna(0)
    elif fill_method == "mean":
        df = df.fillna(df.mean(numeric_only=True))
    elif fill_method == "median":
        df = df.fillna(df.median(numeric_only=True))
    elif fill_method == "drop":
        df = df.dropna()
    print("Final dtypes:", df.dtypes)
    print("Sample data:", df.head())
    return df

def load_data():
    global data
    try:
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx;*.xls")])
        if not file_path:
            return
        progress["value"] = 20
        status_label.config(text="Loading file...")
        root.update_idletasks()
        sheet_names = pd.ExcelFile(file_path).sheet_names
        if len(sheet_names) == 1:
            selected_sheet = sheet_names[0]
            header_row = detect_header_row(file_path, selected_sheet)
            data = pd.read_excel(file_path, sheet_name=selected_sheet, header=header_row)
            data = process_data(data, fill_method=fill_var.get())
            update_tabs()
        else:
            sheet_selection_window = ttk.Toplevel(root)
            sheet_selection_window.title("Select Sheet and Options")
            sheet_selection_window.geometry("400x250")
            sheet_selection_window.transient(root)
            sheet_selection_window.grab_set()
            ttk.Label(sheet_selection_window, text="Select a sheet:", font=("Helvetica", 12)).pack(pady=10)
            sheet_dropdown = ttk.Combobox(sheet_selection_window, values=sheet_names, state="readonly", bootstyle=PRIMARY)
            sheet_dropdown.pack(pady=5)
            sheet_dropdown.current(0)
            ttk.Label(sheet_selection_window, text="Handle missing values:").pack(pady=5)
            fill_dropdown = ttk.Combobox(sheet_selection_window, values=["zero", "mean", "median", "drop"], state="readonly")
            fill_dropdown.pack(pady=5)
            fill_dropdown.current(0)
            def confirm_selection():
                global data
                selected_sheet = sheet_dropdown.get()
                header_row = detect_header_row(file_path, selected_sheet)
                try:
                    data = pd.read_excel(file_path, sheet_name=selected_sheet, header=header_row)
                    data = process_data(data, fill_method=fill_dropdown.get())
                    update_tabs()
                    sheet_selection_window.destroy()
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load sheet '{selected_sheet}': {str(e)}")
            confirm_button = ttk.Button(sheet_selection_window, text="Load Sheet", command=confirm_selection, bootstyle=SUCCESS)
            confirm_button.pack(pady=10)
        progress["value"] = 100
        status_label.config(text="Data loaded successfully.")
    except Exception as e:
        progress["value"] = 0
        status_label.config(text="Error loading file.")
        messagebox.showerror("Error", f"Failed to load file: {str(e)}")

def update_tabs():
    single_tab.update_dropdowns(data)
    multiple_tab.update_dropdowns(data)
    data_viewer_tab.update_table(data)
    clustering_tab.update_dropdowns(data)

def clear_all():
    single_tab.clear_plot()
    multiple_tab.clear_results()
    clustering_tab.clear_plot()
    data_viewer_tab.update_table(data)
    status_label.config(text="All results cleared.")

def save_plot():
    if notebook.index(notebook.select()) == 3:
        file_path = filedialog.asksaveasfilename(defaultextension=".html", filetypes=[("HTML files", "*.html")])
        if file_path:
            clustering_tab.save_plot(file_path)
            messagebox.showinfo("Success", "Plot saved successfully as HTML.")
    else:
        messagebox.showinfo("Info", "Plot saving is only available in the Clustering tab as HTML.")

def show_string_mappings():
    global string_mappings, data
    if not string_mappings:
        messagebox.showinfo("Info", "No string mappings available.")
        return
    mapping_window = ttk.Toplevel(root)
    mapping_window.title("String to Number Mappings")
    mapping_window.geometry("600x400")
    mapping_window.transient(root)
    mapping_window.grab_set()
    tree = ttk.Treeview(mapping_window, columns=("Column", "String", "Number"), show="headings")
    tree.heading("Column", text="Column Name")
    tree.heading("String", text="String Value")
    tree.heading("Number", text="Numerical Value")
    tree.column("Column", width=150)
    tree.column("String", width=200)
    tree.column("Number", width=100)
    tree.pack(fill="both", expand=True, padx=10, pady=10)
    scrollbar = ttk.Scrollbar(mapping_window, orient="vertical", command=tree.yview)
    scrollbar.pack(side="right", fill="y")
    tree.configure(yscrollcommand=scrollbar.set)
    entries = {}
    for col, mapping in string_mappings.items():
        for string, num in mapping.items():
            iid = tree.insert("", "end", values=(col, string, num))
            entries[iid] = (col, string)
    def edit_number(event):
        item = tree.selection()
        if item:
            iid = item[0]
            col, string = entries[iid]
            current_num = tree.item(iid, "values")[2]
            edit_window = ttk.Toplevel(mapping_window)
            edit_window.title("Edit Number")
            edit_window.geometry("250x150")
            edit_window.transient(mapping_window)
            edit_window.grab_set()
            ttk.Label(edit_window, text=f"Column: {col}").pack(pady=5)
            ttk.Label(edit_window, text=f"String: {string}").pack(pady=5)
            num_entry = ttk.Entry(edit_window)
            num_entry.insert(0, current_num)
            num_entry.pack(pady=5)
            def save_edit():
                try:
                    new_num = int(num_entry.get())
                    string_mappings[col][string] = new_num
                    tree.item(iid, values=(col, string, new_num))
                    if col in string_mappings:
                        data[col] = data[col].map(string_mappings[col])
                    update_tabs()
                    edit_window.destroy()
                except ValueError:
                    messagebox.showerror("Error", "Please enter a valid integer.")
            ttk.Button(edit_window, text="Save", command=save_edit, bootstyle=SUCCESS).pack(pady=5)
    tree.bind("<Double-1>", edit_number)

def on_closing():
    root.destroy()
    sys.exit()

root = ttk.Window(themename="superhero")
root.title("Regression Analysis Tool")
root.geometry("1200x700")
root.minsize(1000, 600)
root.protocol("WM_DELETE_WINDOW", on_closing)

main_frame = ttk.Frame(root)
main_frame.pack(fill="both", expand=True)

header = ttk.Label(main_frame, text="Regression Analysis Tool", font=("Helvetica", 18, "bold"), bootstyle=PRIMARY)
header.grid(row=0, column=0, columnspan=3, pady=10)

button_frame = ttk.Frame(main_frame)
button_frame.grid(row=1, column=0, columnspan=3, pady=10)
load_button = ttk.Button(button_frame, text="Load Excel File", command=load_data, bootstyle=INFO)
load_button.pack(side="left", padx=5)
clear_button = ttk.Button(button_frame, text="Clear All", command=clear_all, bootstyle=WARNING)
clear_button.pack(side="left", padx=5)
save_plot_button = ttk.Button(button_frame, text="Save Plot", command=save_plot, bootstyle=SUCCESS)
save_plot_button.pack(side="left", padx=5)
mapping_button = ttk.Button(button_frame, text="String Mappings", command=show_string_mappings, bootstyle=PRIMARY)
mapping_button.pack(side="left", padx=5)

fill_var = tk.StringVar(value="zero")
fill_label = ttk.Label(button_frame, text="Fill NaN:")
fill_label.pack(side="left", padx=5)
fill_dropdown = ttk.Combobox(button_frame, textvariable=fill_var, values=["zero", "mean", "median", "drop"], state="readonly")
fill_dropdown.pack(side="left", padx=5)

notebook = ttk.Notebook(main_frame, bootstyle=SECONDARY)
notebook.grid(row=2, column=0, columnspan=3, sticky="nsew", padx=10, pady=10)

single_tab = SingleRegressionTab(notebook)
notebook.add(single_tab.frame, text="Single Regression")
multiple_tab = MultipleRegressionTab(notebook)
notebook.add(multiple_tab.frame, text="Multiple Regression")
data_viewer_tab = DataViewerTab(notebook)
notebook.add(data_viewer_tab.frame, text="Data Viewer")
clustering_tab = ClusteringTab(notebook)
notebook.add(clustering_tab.frame, text="Clustering")

main_frame.grid_rowconfigure(2, weight=1)
main_frame.grid_columnconfigure(0, weight=1)

status_frame = ttk.Frame(main_frame)
status_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=5)
progress = ttk.Progressbar(status_frame, length=200, mode="determinate")
progress.pack(side="left", padx=5)
status_label = ttk.Label(status_frame, text="Ready", font=("Arial", 10))
status_label.pack(side="left", padx=5)

root.mainloop()