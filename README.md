**PyQt5 Analytics Program** WORK IN PROGRESS

This is a desktop application built with PyQt5 that provides a comprehensive set of tools for data analysis and visualization. It includes features like data preprocessing, time series analysis, machine learning, statistical tests, clustering, regression analysis, and interactive data visualization using Plotly. Designed for analysts, researchers, and data enthusiasts, this program offers an intuitive graphical interface to explore and analyze data efficiently.

---

Installation

To run this program, you need to have Python (version 3.6 or higher) installed on your system. Follow these steps to set up the environment and install the required dependencies:

1. Clone the repository:
   git clone https://github.com/your-username/pyqt5-analytics-program.git
   cd pyqt5-analytics-program

2. Install dependencies using pip:
   pip install pyqt5 pandas numpy statsmodels sklearn plotly reportlab cryptography pyqtwebengine

   These libraries provide the core functionality:
   - pyqt5: For the graphical user interface.
   - pandas and numpy: For data manipulation and numerical operations.
   - statsmodels: For statistical modeling (e.g., ARIMA).
   - sklearn: For machine learning and clustering.
   - plotly: For interactive visualizations.
   - reportlab: For generating PDF reports.
   - cryptography: For basic data encryption.
   - pyqtwebengine: For rendering Plotly visualizations.

3. Ensure you have a working Python environment (e.g., a virtual environment is recommended).

---

Usage

To launch the program, navigate to the repository directory and run the main script:

   python qt_mainregress.py

This will open a window with multiple tabs, each offering different analytical tools. You can load data, preprocess it, perform analyses, and generate reports directly from the interface.

---

Features

The PyQt5 Analytics Program offers a rich set of features:

- Data Preprocessing: Handle missing values, scale data, and apply transformations.
- Time Series Analysis: Perform ARIMA modeling and visualize time series data.
- Machine Learning: Train decision tree classifiers and evaluate their performance.
- Statistical Tests: Conduct T-tests and Chi-square tests on your data.
- Clustering: Apply K-Means clustering and visualize the results.
- Regression Analysis: Perform single and multiple linear regression with outlier exclusion options.
- Data Visualization: Create interactive plots using Plotly for data exploration.
- Report Generation: Generate PDF reports summarizing your analysis.
- Undo/Redo: Revert or redo preprocessing steps as needed.
- Security: Encrypt your data with basic encryption for added security.

---

Data Loading

The program supports loading data from multiple sources:
- CSV files
- Excel files
- SQL databases

To load data:
1. Open the Data Viewer tab.
2. Select your file or database connection.
3. The data will be displayed in a table format, ready for preprocessing or analysis.

Ensure your data is properly formatted (e.g., headers in the first row for CSV/Excel files).

---

Preprocessing

The Data Preprocessing tab provides tools to clean and prepare your data:
- Handle Missing Values: Drop rows with missing data or fill them with the mean.
- Scale Data: Apply standardization (zero mean, unit variance) or normalization (min-max scaling).
- Transformations: Additional options for data transformation are available.

Changes are tracked, allowing you to undo or redo steps (see the Undo/Redo section).

---

Analysis Tabs

The program includes several tabs for different types of analysis:

- Time Series Analysis:  
  Select time and value columns to run ARIMA models and visualize forecasts.

- Machine Learning:  
  Choose a target variable and features to train a decision tree classifier, with performance metrics provided.

- Statistical Tests:  
  Perform T-tests or Chi-square tests by selecting variables from your dataset.

- Clustering:  
  Apply K-Means clustering to selected features and visualize the resulting clusters.

- Single Regression:  
  Conduct linear regression with one independent variable, with options to exclude outliers.

- Multiple Regression:  
  Perform linear regression with multiple independent variables and review detailed results.

- Data Viewer:  
  View, filter, and explore your loaded data in a table format.

- Correlation Heatmap:  
  Generate a heatmap to visualize correlation matrices for selected variables.

Each tab provides an intuitive interface to configure and execute the analysis.

---

Visualization

The program leverages Plotly to create interactive visualizations. Features include:
- Zooming and panning to explore data.
- Hovering over data points to view details.
- Exporting plots for use in reports or presentations.

Visualizations are available across all analysis tabs where applicable.

---

Report Generation

Generate professional PDF reports with ease:
1. Perform your desired analyses.
2. Click the Generate Report button.
3. The report will include a summary of your data and analysis results.

Reports are saved locally and can be shared or archived.

---

Undo/Redo

The program maintains a history of preprocessing steps. Use the undo and redo buttons to:
- Revert unwanted changes.
- Reapply steps you’ve undone.

This feature ensures flexibility during data preparation.

---


Contributing

We welcome contributions to enhance the PyQt5 Analytics Program! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   git checkout -b feature/your-feature-name
3. Commit your changes and push to your fork.
4. Submit a pull request with a clear description of your changes.


