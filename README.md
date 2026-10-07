# Telco Customer Churn Prediction (AA DSS110)

Case study on the IBM Telco Customer Churn dataset: supervised binary classification of whether a customer churns.

| File | Purpose |
|---|---|
| `Telco_Churn_Case_Study.ipynb` | Executed notebook (EDA, modelling, evaluation, prediction) |
| `Telco_Churn_Step_by_Step_Code.py` | Same code as VS Code cells (`# %%`), step by step |
| `Telco_Churn_Case_Study.pdf` | PDF documentation of the case study |
| `Project Proposal_ Telco Customer Churn Prediction.pdf` | Project proposal |
| `Telco-Customer-Churn.csv` | Dataset (IBM sample data, also on Kaggle: blastchar/telco-customer-churn) |
| `figures/`, `model_results.csv` | Charts and metrics produced by the notebook |

## Run
```
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
jupyter notebook Telco_Churn_Case_Study.ipynb
```

## Results (test set, 20% stratified split)
Random Forest, Gradient Boosting and Logistic Regression reach ROC-AUC 0.842-0.846 and recall of about 0.79-0.80 on churners, with precision of about 0.51-0.52.
