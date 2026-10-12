# 🌫️ Delhi-NCR AQI Intelligence

**An interactive Streamlit application for air quality analysis and short-term Air Quality Index (AQI) forecasting across Delhi-NCR using Python and Machine Learning.**

Explore historical air quality trends, compare locations, analyze pollutant distributions, inspect data quality, and evaluate machine learning models for short-term AQI forecasting.

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Dataset](#-dataset)
- [Application Modules](#-application-modules)
- [Machine Learning Methodology](#-machine-learning-methodology)
- [Evaluation Metrics](#-evaluation-metrics)
- [Project Structure](#-project-structure)
- [Installation and Setup](#-installation-and-setup)
- [Usage](#-usage)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

## 🎯 Overview

Air pollution is a major environmental concern in Delhi-NCR. Understanding historical pollution patterns and estimating future AQI values can help people interpret air quality conditions and support data-driven environmental analysis.

**Delhi-NCR AQI Intelligence** is a Python-based web application built with Streamlit. It provides an interactive interface for exploring historical air quality data and evaluating short-term forecasting models.

The application supports dataset loading through Kaggle or user-uploaded CSV/ZIP files. It provides visual analytics, city-wise comparisons, pollutant analysis, data quality checks, and optional machine learning evaluation.

### Project Objectives

- Analyze historical AQI patterns across locations.
- Compare air quality across different cities or monitoring locations.
- Investigate pollutant distributions and correlations.
- Identify potential data quality issues.
- Evaluate short-term AQI forecasting using machine learning.
- Compare XGBoost and Ridge Regression against simple forecasting baselines.

## ✨ Key Features

### 📊 Interactive AQI Dashboard
- View key air quality statistics.
- Explore AQI trends over time.
- Filter data by location and date range.
- Compare air quality patterns across selected locations.

### 🏙️ Location-Wise Analysis
- Compare AQI across available locations.
- Explore differences in pollution levels.
- Investigate monthly and hourly air quality patterns.

### 🧪 Pollutant Analysis
Analyze available pollutant measurements, including:

- PM2.5 — Fine particulate matter
- PM10 — Coarse particulate matter
- NO2 — Nitrogen dioxide
- SO2 — Sulfur dioxide
- CO — Carbon monoxide
- O3 — Ozone

The availability of individual pollutant columns depends on the uploaded dataset.

### 📈 Exploratory Data Analysis
- Historical AQI trend visualization.
- Monthly and hourly pattern analysis.
- Pollutant distribution analysis.
- Correlation analysis between numerical variables.

### 🧹 Data Quality Checks
- Inspect missing or invalid values.
- Examine the available dataset structure.
- Filter data interactively.
- Download filtered data as a CSV file.

### 🤖 Machine Learning Evaluation
The application includes optional short-term forecasting evaluation using:

- XGBoost Regressor
- Ridge Regression
- Persistence baseline
- Month-hour climatology baseline

The current implementation evaluates **1-hour-ahead and 24-hour-ahead forecasts** using a historical test period.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical computations |
| Matplotlib / Plotting libraries | Data visualization, where used |
| Scikit-learn | Ridge Regression and evaluation metrics |
| XGBoost | Gradient-boosted tree regression |
| KaggleHub | Dataset download from Kaggle |

## 📂 Dataset

The application uses the following publicly available Kaggle dataset:

**Delhi NCR Hourly Air Quality Dataset (2019–2024)**

Dataset: https://www.kaggle.com/datasets/aniket0712/delhi-ncr-hourly-air-quality-dataset-20192024

### Data Source Options

The application supports two ways to load data:

1. **Download from Kaggle:** Download the dataset through KaggleHub.
2. **Upload CSV / ZIP:** Upload a compatible dataset through the application sidebar.

### Expected Data Columns

The application expects columns corresponding to the following fields:

| Column | Description |
|---|---|
| `timestamp` | Date and time of the observation |
| `location_name` | Name of the monitoring location |
| `aqi` | Air Quality Index |
| `pm25` | PM2.5 concentration |
| `pm10` | PM10 concentration |
| `no2` | Nitrogen dioxide concentration |
| `so2` | Sulfur dioxide concentration |
| `co` | Carbon monoxide concentration |
| `o3` | Ozone concentration |

The timestamp, location, and AQI fields are central to the application. Pollutant columns are used when available.

**Note:** Actual column names, time coverage, missing values, and the number of records depend on the dataset version loaded.

## 🧩 Application Modules

### 1. Data Loading
- Load data from Kaggle or an uploaded CSV/ZIP archive.
- Parse timestamps and standardize the data structure.
- Validate the essential columns before analysis.

### 2. Data Filtering
- Select one or more locations.
- Choose a date range.
- Inspect the filtered observations.
- Export filtered data when required.

### 3. AQI Analysis
- Explore AQI statistics and historical trends.
- Compare locations.
- Analyze monthly and hourly patterns.

### 4. Pollutant Analysis
- Visualize pollutant distributions.
- Explore numerical correlations.
- Investigate relationships between pollutants and AQI.

### 5. Data Quality Analysis
- Review data completeness and quality.
- Identify potential issues before modeling.
- Examine the data used for downstream analysis.

### 6. Forecasting and Model Evaluation
- Construct short-term forecasting features from historical data.
- Evaluate Ridge Regression and XGBoost.
- Compare model predictions against simple baselines.
- Report forecasting metrics on the held-out test period.

## 🧠 Machine Learning Methodology

The forecasting component treats AQI prediction as a supervised regression problem.

### Forecasting Horizons

| Horizon | Objective |
|---|---|
| 1 hour | Estimate AQI one hour ahead |
| 24 hours | Estimate AQI 24 hours ahead |

### Models

**XGBoost Regressor**

XGBoost uses gradient-boosted decision trees to model nonlinear relationships between input features and the target variable.

**Ridge Regression**

Ridge Regression is a linear regression model with L2 regularization. It provides a simpler reference model for comparison.

**Persistence Baseline**

The persistence baseline assumes the future AQI will equal the latest available AQI.

**Month-Hour Climatology Baseline**

This baseline estimates AQI using historical averages associated with the corresponding month and hour.

### Time-Based Evaluation

The current implementation uses a chronological evaluation setup rather than randomly shuffling time-series observations:

- 2019–2021: Model fitting period
- 2022: Validation-period reference
- 2023: Held-out test period

The exact usable records depend on data availability and the selected forecasting horizon.

The application uses a compact XGBoost configuration to keep interactive evaluation practical.

## 📏 Evaluation Metrics

The application reports the following metrics:

| Metric | Meaning |
|---|---|
| MAE | Mean Absolute Error; average absolute prediction error |
| RMSE | Root Mean Squared Error; gives greater weight to large errors |
| R² | Coefficient of determination; measures performance relative to the target's variation |

Lower MAE and RMSE values generally indicate better predictions. R² should be interpreted alongside the error metrics and forecasting baselines.

No model is guaranteed to outperform the baselines; actual results depend on the dataset and test period.

## 📁 Project Structure

```text
delhi-ncr-aqi-prediction/
│
├── app.py             # Streamlit application
├── requirements.txt   # Python dependencies
├── README.md          # Project documentation
├── .gitignore         # Files excluded from Git
│
└── .venv/             # Local virtual environment (not committed)
```

The `.venv/` directory is created locally during setup and should not be uploaded to GitHub.

## ⚙️ Installation and Setup

### Prerequisites

- Python installed on your system
- Git
- Internet connection for installing dependencies and downloading data
- Kaggle access and credentials if using the Kaggle download option

### Step 1: Clone the Repository

```bash
git clone https://github.com/gauravbhardwaj55110/Delhi-NCR-AQI-prediction-.git
```

Navigate into the project directory:

```bash
cd Delhi-NCR-AQI-prediction-
```

### Step 2: Create a Virtual Environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 4: Launch the Application

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal, typically:

```text
http://localhost:8501
```

Open that address in your browser.

### Step 5: Load a Dataset

Choose one of the available data source options in the sidebar:

- **Download from Kaggle**, or
- **Upload CSV / ZIP**.

If Kaggle authentication is required, configure your Kaggle credentials according to the official Kaggle documentation, or upload the dataset manually.

## 🚀 Usage

1. Start the application using `streamlit run app.py`.
2. Select the dataset source.
3. Load the dataset.
4. Select the locations you want to analyze.
5. Choose the required date range.
6. Explore AQI trends and pollutant analysis.
7. Inspect data quality and download filtered data if needed.
8. Run the optional forecasting evaluation.
9. Compare model performance using MAE, RMSE, and R².

## ⚠️ Limitations

- Forecasting results depend on the quality and coverage of the selected dataset.
- The forecasting component currently evaluates 1-hour and 24-hour horizons.
- The model evaluation uses historical data; predictions are not live AQI readings.
- The current dataset schema does not include meteorological features, so the application is **not a weather-aware forecasting system**.
- Model evaluation may take time depending on the dataset size and hardware.
- Results should be interpreted as analytical estimates, not as official air quality measurements or health advice.
- Kaggle downloads may require valid credentials and network access.

## 🔮 Future Improvements

Potential extensions include:

- Add LSTM models for time-series forecasting.
- Evaluate additional forecast horizons, such as 3, 6, and 12 hours.
- Integrate meteorological variables such as temperature, humidity, pressure, and wind speed when reliable data is available.
- Add feature importance and SHAP-based model explainability.
- Improve hyperparameter tuning and model comparison.
- Add more robust time-series cross-validation.
- Deploy the application for public access.
- Add automated data updates and monitoring.

These are planned enhancements and are not all implemented in the current application.

## 👨‍💻 Author

**Gaurav Bharadwaj**

GitHub: https://github.com/gauravbhardwaj55110

Project Repository: https://github.com/gauravbhardwaj55110/Delhi-NCR-AQI-prediction-

## 📄 License

No license has been specified yet. If you want others to reuse, modify, and distribute this project, consider adding an appropriate open-source license.

---

⭐ If you find this project useful, consider giving the repository a star!
