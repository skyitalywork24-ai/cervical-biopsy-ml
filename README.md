# Cervical Cancer Biopsy Prediction – ML Assignment 2

## 1. Problem Statement
Predict whether a patient's cervical biopsy result is positive (`Biopsy` = 1) from
demographic, lifestyle and medical-history risk factors, using five classification
models, and compare them with Accuracy, AUC, Precision, Recall, F1 and MCC.
A Streamlit app lets a user upload test data and evaluate any saved model.

## 2. Dataset Description
- **Source:** UCI Machine Learning Repository – *Cervical Cancer (Risk Factors)*
- **Instances:** 858 patients  |  **Features used:** 32 (after removing 3 leaky columns)
- **Target:** `Biopsy` (0 = negative, 1 = positive) – binary, strongly imbalanced (~6.4% positive)
- **Features:** age, number of sexual partners, age at first intercourse, pregnancies,
  smoking (flag/years/packs per year), hormonal contraceptives, IUD, STD history
  (per-STD flags, counts, time since diagnosis), and prior diagnoses (Dx:Cancer, Dx:CIN, Dx:HPV, Dx).
- **Preprocessing:** `?` treated as missing; median imputation and standard scaling inside each
  scikit-learn Pipeline (fit on training data only); stratified 75/25 train/test split (seed 42).
- **Excluded columns:** `Hinselmann`, `Schiller`, `Citology` are other screening tests for the same
  disease and would leak the answer into the model.
- **Imbalance handling:** `class_weight="balanced"` for Logistic Regression, Decision Tree and SVM.

## 3. GitHub Repository Link
<PASTE YOUR REPO LINK HERE>

Live app: <PASTE YOUR STREAMLIT LINK HERE>

## 4. Models Used – Comparison Table
(Paste values from `model/metrics.csv` after running `train_models.py`)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Support Vector Machine | | | | | | |

## 5. Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | |
| Decision Tree | |
| KNN | |
| Naive Bayes | |
| Support Vector Machine | |
| **Overall, Winner for your dataset?** | |

## How to run
```bash
pip install -r requirements.txt
python train_models.py risk_factors_cervical_cancer.csv   # trains + saves models + test_data.csv
streamlit run app.py
```
