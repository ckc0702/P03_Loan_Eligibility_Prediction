# P03_Loan_Eligibility_Prediction

This project contain three machine learning algorithms to predict the eligibility of a person for loan. There are two main category of problems that loan eligibility can be categorize to: 

1. Regression problem. When the goal is to predict the maximum amount of loan a person can borrow.
2. Classification problem. When the goal is to predict if a person can sucessfully borrow the loan.

This project focus on the classification aspect, hence three popular machine learning algorithms for classification were applied, namely, logistics regression, decision tree, and random forest.

Dataset: https://www.kaggle.com/datasets/yasserh/loan-default-dataset

Random Forest (default param): 0.9039 (Accuracy) | 0.8571 (Specificity) 
Decision Tree Accuracy (default param): 0.8295 (Accuracy) | 0.6582 (Specificity)
Logistics Regression Accuracy (default param): 0.8302 (Accuracy) | 0.6586 (Specificity)

Random Forest (tuned): 0.9043 (Accuracy) | 0.8577 (Specificity) 
Decision Tree Accuracy (tuned): 0.8517 (Accuracy) | 0.6903 (Specificity)
Logistics Regression [NO TUNABLE PARAM]
