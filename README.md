# Insurance EDA and Cost Prediction

An end-to-end data analytics and machine-learning project using a medical insurance dataset. The project explores the factors associated with insurance charges and trains regression models to estimate individual medical costs.

## Project Objectives

- Inspect data quality, missing values, duplicates, distributions, and outliers.
- Analyse relationships between age, BMI, smoking status, region, children, and charges.
- Prepare numerical and categorical features using a scikit-learn pipeline.
- Compare Linear Regression and Random Forest Regression.
- Save the best-performing model for reusable predictions.

## Dataset

The dataset contains 1,338 records and seven columns:

| Column | Description |
|---|---|
| `age` | Age of the insured person |
| `sex` | Gender |
| `bmi` | Body mass index |
| `children` | Number of dependent children |
| `smoker` | Smoking status |
| `region` | Residential region |
| `charges` | Medical insurance cost |

## Repository Structure

```text
Insurance-EDA-and-Cost-Prediction/
├── data/
│   └── insurance.csv
├── notebooks/
│   └── insurance_eda.ipynb
├── src/
│   ├── train_model.py
│   └── predict.py
├── models/
│   └── insurance_cost_model.joblib
├── reports/
│   └── model_metrics.json
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run the EDA Notebook

```bash
jupyter notebook notebooks/insurance_eda.ipynb
```

## Train the Models

```bash
python src/train_model.py
```

The script evaluates both models using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² score

The best model is saved to `models/insurance_cost_model.joblib`.

## Run a Sample Prediction

```bash
python src/predict.py
```

## Main Findings

- Smoking status is the strongest categorical driver of higher medical charges.
- Age generally has a positive association with charges.
- Higher BMI can increase predicted costs, especially for smokers.
- The target variable contains genuine high-cost observations, so outlier removal should be performed only with domain justification.

## Technologies

Python, Pandas, NumPy, Matplotlib, Seaborn, SciPy, scikit-learn, Jupyter Notebook, Joblib.

## Author

**Mousom Koley**
