# Credit Card Fraud Detection System Using Machine Learning

Minor project (BCA). Compares Logistic Regression, Decision Tree and Random Forest on an
imbalanced transaction dataset and provides a small Flask demo.

## 1. Setup
```
pip install -r requirements.txt
```

## 2. Dataset
Download `creditcard.csv` (Kaggle: "Credit Card Fraud Detection" by MLG-ULB) and place it at:
```
data/creditcard.csv
```
Columns: Time, V1..V28, Amount, Class (1 = fraud).

## 3. Train and evaluate
```
python train.py --data data/creditcard.csv
```
Optional: add `--smote` to also test SMOTE. For a quick test without the dataset: `python train.py --demo`.
Random Forest on the full dataset can take a few minutes.

Created in `outputs/`: model_comparison.csv, threshold_analysis.csv/.png, pr_curves.png,
confusion_matrix.png, feature_importance.csv/.png, final_report.json.
Created in `models/`: best_model.joblib, sample_transactions.json.

## 4. Run the demo
```
python app.py
```
Open http://127.0.0.1:5000 and click a sample button, then Predict.
The demo samples are the fraud cases the model is most confident about (for a clean demonstration)
plus some legitimate cases - mention this in your report; use real test results from `outputs/` for evaluation.

## Method (matches synopsis)
- Duplicates/missing values removed; Time and Amount scaled (fit on training data only)
- Stratified split: 64% train, 16% validation, 20% test
- Imbalance handling: original distribution vs class weights (optional SMOTE on training data only)
- Metrics: accuracy, precision, recall, F1, ROC-AUC, AUC-PR, confusion matrix
- Best model chosen on the validation set (AUC-PR); decision threshold also chosen on the validation set
- Threshold analysis on the test set; Random Forest feature importance
- Limitation: single dataset, anonymised features (V1-V28), no concept-drift or real-time testing
