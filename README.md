# Dragon Real Estate Price Predictor

A machine learning project that predicts housing prices using the Boston Housing dataset. The project compares three regression models — **Linear Regression**, **Decision Tree**, and **Random Forest** — and selects the best performer for deployment.

---

##  Model Performance (Cross-Validation RMSE)

| Model | Mean RMSE | Std Deviation |
|---|---|---|
| Decision Tree | 4.19 | 0.85 |
| Linear Regression | 4.22 | 0.75 |
| **Random Forest** | **3.49** | **0.76** |

> **Random Forest** was selected as the final model due to its lowest RMSE and good generalization.


##  How It Works

### 1. Data Loading & Exploration
- Loads the housing dataset from an Excel file
- Plots histograms for visual exploration
- Checks value distributions for the `CHAS` column (river proximity)

### 2. Stratified Train/Test Split
- Splits the data 80/20 while preserving the proportion of `CHAS` values in both sets

### 3. Feature Engineering
- Creates a new feature: `TAXRM = TAX / RM` (tax rate per room)
- Generates a scatter matrix to visualize correlations

### 4. Preprocessing Pipeline
- Handles missing values using **median imputation**
- Standardizes features using **StandardScaler**

### 5. Model Training & Evaluation
- Trains and compares Linear Regression, Decision Tree, and Random Forest
- Uses **10-fold cross-validation** to detect overfitting
- Evaluates final model on the held-out test set

### 6. Model Saving & Loading
- Saves the trained model using `joblib`
- Automatically loads the saved model if it already exists

---

##  Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib openpyxl joblib
```

### Run the Project

```bash
python dragon_real_estate.py
```

Make sure `ml.xlsx` is available and update its path in the script if needed:
```python
housing = pd.read_excel("path/to/ml.xlsx")
```

---

##  Dataset Features

| Feature | Description |
|---|---|
| `CRIM` | Per capita crime rate |
| `ZN` | Proportion of residential land |
| `INDUS` | Proportion of non-retail business acres |
| `CHAS` | Charles River proximity (1 = yes, 0 = no) |
| `NOX` | Nitric oxide concentration |
| `RM` | Average number of rooms per dwelling |
| `AGE` | Proportion of old owner-occupied units |
| `DIS` | Distances to employment centres |
| `RAD` | Accessibility to highways |
| `TAX` | Property tax rate |
| `PTRATIO` | Pupil-teacher ratio |
| `B` | Proportion of residents of African American descent |
| `LSTAT` | % lower status of the population |
| `MEDV` | **Target** — Median home value (in $1000s) |
| `TAXRM` | **Engineered** — TAX / RM ratio |

---

##  Key Observations

- **Linear Regression** underfits slightly (RMSE ~4.22)
- **Decision Tree** overfits on training data (RMSE = 0 on train), but generalizes poorly
- **Random Forest** balances bias and variance best, achieving the lowest cross-validation RMSE (~3.49)

---

## 👤 Author

Built as part of a machine learning learning project using scikit-learn.
