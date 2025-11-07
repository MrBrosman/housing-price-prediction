# 🏠 Housing Price Prediction — Ridge Regression Model (King County Dataset)

### Author: Bosco Rosillo  
Project developed as part of an academic Machine Learning project focused on **regression modeling**, **feature engineering**, and **model interpretability** using Python and Scikit-Learn.

---

## 📘 Project Overview

The goal of this project is to build a **robust and interpretable regression model** to predict housing prices based on structural, locational, and temporal features.  
This end-to-end analysis covers **data preprocessing**, **EDA**, **feature engineering**, **regularized regression**, and **model deployment**.

The final model — a **Ridge Regression with log-transformed target** — achieves **R² = 0.94** on test data, showing excellent explanatory power and stable generalization.

---

## 🧾 About the Data

This dataset comprises **one year of house sale prices (May 2014 – May 2015)** across **King County**, including **Seattle**.  
It contains **21 features** describing property characteristics, location, and condition — offering a detailed snapshot of the regional real-estate market.

### Dataset Features
| Column | Description |
|:--|:--|
| `id` | Unique identifier for a house |
| `date` | Date on which the house was sold |
| `price` | Sale price of the house *(prediction target)* |
| `bedrooms` | Number of bedrooms in the house |
| `bathrooms` | Number of bathrooms (including partials) |
| `sqft_living` | Interior living space in square feet |
| `sqft_lot` | Land space in square feet |
| `floors` | Number of levels in the house |
| `waterfront` | Whether the property has a waterfront view |
| `view` | Number of times the house has been viewed |
| `condition` | Overall physical condition of the property |
| `grade` | Overall construction and design grade (King County scale) |
| `sqft_above` | Square footage above the basement |
| `sqft_basement` | Square footage of the basement |
| `yr_built` | Year the house was originally built |
| `yr_renovated` | Year of last renovation (0 = never renovated) |
| `zipcode` | ZIP code of the property |
| `lat` | Latitude coordinate |
| `long` | Longitude coordinate |
| `sqft_living15` | Living area of the nearest 15 neighboring properties (2015) |
| `sqft_lot15` | Lot area of the nearest 15 neighboring properties (2015) |

**Target → `price`:**  
The main goal is to identify which features most strongly influence housing prices and build a predictive model to estimate them.  
Additional focus is placed on properties above **\$650 K**, representing the upper-tier market.

### 📚 Data Source
Dataset publicly available on Kaggle:  
[King County Houses (AA) — Minas Ameh](https://www.kaggle.com/datasets/minasameh55/king-country-houses-aa)

---

## ⚙️ Methodology

### 1. Exploratory Data Analysis (EDA)
- Confirmed absence of missing values and validated data types.  
- Examined distributions and correlations (Pearson matrix).  
- Removed 13 invalid records (0 bedrooms).  
- Identified key positive correlations with price:  
  `sqft_living (0.70)`, `grade (0.67)`, `bathrooms (0.53)`, `view (0.40)`.

### 2. Feature Engineering
- Extracted `sale_year` and `sale_month` from the sale date.  
- Derived `house_age`, `years_since_renovation`, and `price_per_sqft`.  
- Created binary `renovated_flag` (1 = renovated).  
- Implemented standardization and one-hot encoding via `ColumnTransformer`.

### 3. Model Selection and Regularization
| Model | Regularization | R² (CV) | Comment |
|:--|:--|:--:|:--|
| Linear Regression | None | 0.914 | Baseline with overfitting |
| Ridge | L2 | **0.951** | Stable and best CV performance |
| Lasso | L1 | 0.602 | Sparse but weaker fit |
| ElasticNet | L1 + L2 | 0.926 | Balanced but poor test generalization |

Selected **Ridge Regression (α = 0.1)** for optimal bias–variance trade-off.

### 4. Outlier Capping + Log Transformation
- **Capped prices above 99th percentile (≈ \$1.97 M)** — 216 observations.  
- Applied `np.log1p()` to target for variance stabilization.  
- Achieved homoscedastic, near-normal residuals and improved robustness.

---

## 📈 Model Performance (Log Scale)

| Metric | Value | Interpretation |
|:--|--:|:--|
| **R²** | **0.940** | Explains 94 % of variance |
| **MAE** | 0.087 | ≈ 8.7 % mean prediction error |
| **RMSE** | 0.128 | ≈ 13 % typical deviation |

Residual diagnostics confirm:  
- No heteroscedasticity  
- Centered, symmetric residuals  
- Stable predictions across all price ranges

---

## 🧱 Final Model Deployment

The finalized Ridge pipeline was exported for reuse:

```python
import joblib
model = joblib.load("ridge_log_model_final.joblib")
predictions = model.predict(new_data)

## 📊 Visual Evaluation

### 1. Actual vs Predicted (Log Scale)
- Points align closely around the 45° identity line.  
- Confirms high predictive precision after log transformation.

### 2. Actual vs Predicted (USD Scale)
- Tight clustering around equality line; minimal spread.  
- High-price variance effectively controlled via capping.

### 3. Residuals by Price Decile
| Decile | Mean Residual (USD) | Std (USD) | Interpretation |
|:--:|--:|--:|:--|
| 0 – 3 | −16 k → −5 k | 25 k – 40 k | Slight underestimation (low homes) |
| 4 – 7 | −2 k → +26 k | 50 k – 56 k | Balanced, low bias |
| 8 | +27 k | 92 k | Mild overestimation |
| 9 | −75 k | 990 k | Expected luxury-tier variance |

Residuals remain centered near zero → **no systemic bias**.

---

## 💡 Economic Insights

- **Grade** and **Living Area** are the strongest price drivers.  
- Each additional grade level ≈ +10 – 12 % price increase.  
- **Waterfront** adds ~35 – 40 % premium.  
- **Renovations** significantly boost valuation.  
- **Older homes** depreciate gradually unless renovated or premium location.

These outcomes are fully consistent with real-estate valuation patterns in King County.

---

## 🧾 Deliverables

| File | Description |
|:--|:--|
| `ridge_log_model_final.joblib` | Serialized Ridge model (α = 0.1, log target) |
| `notebook.ipynb` | Full EDA → Modeling → Evaluation workflow |
| Diagnostic Plots | Residuals, actual vs predicted, decile performance |
| `README.md` | Complete project documentation (current file) |

---

## 🧠 Key Takeaways

1. **Regularization + Log Transform:** improved accuracy and model stability.  
2. **Feature Engineering:** temporal + structural variables enhanced predictive power.  
3. **Interpretable ML:** Ridge Regression delivers high performance without black-box complexity.  
4. **Practical Deployment:** ready for integration via API or batch valuation scripts.

---

## 🏁 Final Conclusion

The **Ridge Regression model (α = 0.1)**, trained on log-scaled capped targets, provides:

- **R² = 0.94**  
- **MAE = 0.087**  
- **RMSE = 0.128**

This demonstrates that careful preprocessing and regularization yield **production-ready predictive accuracy** while preserving interpretability.

The model is robust enough for practical property valuation, research, or as a benchmark for advanced ML models (e.g., Random Forest or XGBoost).
