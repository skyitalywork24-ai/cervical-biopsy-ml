# Cervical Cancer Biopsy Prediction – ML Assignment 2

## 1. Problem Statement
Predict whether a patient's cervical biopsy result is positive (`Biopsy` = 1) from
demographic, lifestyle and medical-history risk factors. Five classification models are
trained and compared using Accuracy, AUC, Precision, Recall, F1 and MCC, and deployed in a
Streamlit app where a user can upload test data and evaluate any saved model.

## 2. Dataset Description
- **Source:** UCI Machine Learning Repository – *Cervical Cancer (Risk Factors)*
- **Instances:** 858 patients  |  **Features used:** 32 (after removing 3 leaky columns)
- **Target:** `Biopsy` (0 = negative, 1 = positive). Binary and strongly imbalanced: 803 negative, 55 positive (6.4%).
- **Features:** age, number of sexual partners, age at first intercourse, number of pregnancies,
  smoking (flag, years, packs per year), hormonal contraceptives, IUD, STD history
  (per-STD flags, counts, time since diagnosis) and earlier diagnoses (Dx:Cancer, Dx:CIN, Dx:HPV, Dx).
- **Preprocessing:** `?` treated as missing; median imputation and standard scaling inside each
  scikit-learn Pipeline (fit on training data only); stratified 75/25 split (seed 42), giving
  215 test rows with 14 positives.
- **Excluded columns:** `Hinselmann`, `Schiller` and `Citology` are other screening tests for the
  same disease and would leak the answer into the model.
- **Imbalance handling:** `class_weight="balanced"` for Logistic Regression, Decision Tree and SVM.

## 3. Links
- GitHub repository: https://github.com/skyitalywork24-ai/cervical-biopsy-ml
- Live Streamlit app: https://cervical-biopsy-risk-lab.streamlit.app

## 4. Models Used – Comparison Table
Evaluated on the held-out test split (decision threshold 0.5, positive class = biopsy positive).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8000 | 0.6354 | 0.1628 | 0.5000 | 0.2456 | 0.1979 |
| Decision Tree | 0.7395 | 0.4259 | 0.0625 | 0.2143 | 0.0968 | -0.0057 |
| KNN | 0.9302 | 0.6917 | 0.0000 | 0.0000 | 0.0000 | -0.0180 |
| Naive Bayes | 0.0930 | 0.5149 | 0.0670 | 1.0000 | 0.1256 | 0.0447 |
| Support Vector Machine | 0.9349 | 0.7093 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## 5. Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Most balanced model. Class weighting lets it catch half of the positive cases (Recall 0.50) and gives the best F1 (0.246) and MCC (0.198), at the cost of many false alarms (Precision 0.16). AUC of 0.64 shows only modest ranking ability. |
| Decision Tree | Weak. It finds only 3 of 14 positives, its MCC is about zero, and its AUC (0.43) is below 0.5, so its probability ranking is no better than chance. A depth-limited tree on 640 training rows overfits and does not generalise. |
| KNN | Accuracy of 0.93 is misleading. It predicts almost every patient as negative, so Recall, Precision and F1 are all 0. Distance-based voting is dominated by the majority class, and it has no class-weight option. Its AUC (0.69) shows the neighbour scores still carry some signal. |
| Naive Bayes | The opposite failure: it flags nearly every patient as positive (Recall 1.0) but Accuracy collapses to 0.09 and Precision equals the base rate (0.067). Its independence assumption fits poorly because many features, such as the STD flags, are strongly correlated. |
| Support Vector Machine | Best AUC (0.709), so it ranks patients best, but at the default 0.5 threshold it predicts everyone as negative, giving Accuracy 0.935 (the share of negatives) and zero Recall. Lowering the decision threshold, which the app allows, would be needed to make it useful. |
| **Overall, Winner for your dataset?** | **Logistic Regression** is the best practical model because it is the only one with a positive MCC and a useful Recall/Precision balance. SVM is the best ranker by AUC. Accuracy alone would wrongly favour SVM and KNN. All results rest on a small test set of 14 positives, so they are noisy and should be read with caution. |

## How to run
```bash
pip install -r requirements.txt
python model/train_models.py   # trains, saves models to model/, writes test_data.csv
streamlit run app.py
```
