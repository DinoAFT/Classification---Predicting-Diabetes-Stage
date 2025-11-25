import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# Seperate categorical and numerical features

# Numerical features to be scaled
temp1 = X_train.select_dtypes(include=[np.number])
temp2 = X_test.select_dtypes(include=[np.number])

binary_cols = ['family_history_diabetes','hypertension_history','cardiovascular_history']
X_train_num = temp1.drop(columns=binary_cols)
X_test_num = temp2.drop(columns=binary_cols)

# Categorical features to not be scaled
X_train_cat = X_train.select_dtypes(include=['object', 'category', 'bool'])
X_test_cat = X_test.select_dtypes(include=['object', 'category', 'bool'])
X_train_cat[binary_cols] = X_train[binary_cols]
X_test_cat[binary_cols] = X_test[binary_cols]

# Center and Scale the numerical features

scaler = StandardScaler()

scaler.fit(X_train_num)

X_train_num_std = scaler.transform(X_train_num)
X_test_num_std = scaler.transform(X_test_num)

X_train_num_std = pd.DataFrame(X_train_num_std, columns=X_train_num.columns, index=X_train_num.index)
X_test_num_std = pd.DataFrame(X_test_num_std, columns=X_test_num.columns, index=X_test_num.index)

# Recombine numerical and categorical features

X_train_std = pd.concat([X_train_num_std, X_train_cat], axis=1)
X_test_std = pd.concat([X_test_num_std, X_test_cat], axis=1)

# Export scaled training/test sets as csv files
X_train_std.to_csv('X_train_std.csv')
Y_train.to_csv('Y_train.csv')
X_test_std.to_csv('X_test_std.csv')
Y_test.to_csv('Y_test.csv')
