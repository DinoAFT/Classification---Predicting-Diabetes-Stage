import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

diabetesStage = pd.read_csv("diabetes_dataset.csv")

# Remove id/noise/unused target features
diabetesStage = diabetesStage.drop(['glucose_fasting','hba1c','diabetes_risk_score','diagnosed_diabetes'],axis=1)

# Isolate target and features respectively
Y = diabetesStage.loc[:, 'diabetes_stage']
X = diabetesStage.loc[:, diabetesStage.columns != 'diabetes_stage']

# Split features and target into test and training data
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.25,
    random_state=42
)

# Export training/test sets as csv files
X_train.to_csv('X_train.csv')
Y_train.to_csv('Y_train.csv')
X_test.to_csv('X_test.csv')
Y_test.to_csv('Y_test.csv')
