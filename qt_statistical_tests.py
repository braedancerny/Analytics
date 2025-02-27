from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QComboBox, 
                             QPushButton, QMessageBox, QTextEdit)
import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency

class StatisticalTestsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.data = None

        self.layout = QVBoxLayout(self)
        self.test_label = QLabel("Select Test:")
        self.test_combo = QComboBox()
        self.test_combo.addItems(["T-Test", "Chi-Square Test"])
        self.var1_label = QLabel("Variable 1:")
        self.var1_combo = QComboBox()
        self.var2_label = QLabel("Variable 2:")
        self.var2_combo = QComboBox()
        self.run_button = QPushButton("Run Test")
        self.run_button.clicked.connect(self.run_test)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)

        self.layout.addWidget(self.test_label)
        self.layout.addWidget(self.test_combo)
        self.layout.addWidget(self.var1_label)
        self.layout.addWidget(self.var1_combo)
        self.layout.addWidget(self.var2_label)
        self.layout.addWidget(self.var2_combo)
        self.layout.addWidget(self.run_button)
        self.layout.addWidget(self.results_text)

    def update_dropdowns(self, data: pd.DataFrame):
        self.data = data
        columns = data.columns.tolist()
        self.var1_combo.clear()
        self.var2_combo.clear()
        self.var1_combo.addItems(columns)
        self.var2_combo.addItems(columns)

    def run_test(self):
        if self.data is None:
            QMessageBox.critical(self, "Error", "No data loaded.")
            return
        test = self.test_combo.currentText()
        var1 = self.var1_combo.currentText()
        var2 = self.var2_combo.currentText()
        try:
            if test == "T-Test":
                group1 = self.data[var1]
                group2 = self.data[var2]
                stat, p = ttest_ind(group1, group2)
                result = f"T-Test Result:\nStatistic: {stat:.4f}\nP-value: {p:.4f}"
            elif test == "Chi-Square Test":
                contingency_table = pd.crosstab(self.data[var1], self.data[var2])
                stat, p, dof, expected = chi2_contingency(contingency_table)
                result = f"Chi-Square Test Result:\nStatistic: {stat:.4f}\nP-value: {p:.4f}\nDegrees of Freedom: {dof}"
            self.results_text.setText(result)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Test failed: {str(e)}")
