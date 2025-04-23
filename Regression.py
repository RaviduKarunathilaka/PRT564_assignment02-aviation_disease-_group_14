
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO
from scipy.stats import ttest_ind
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

df = pd.read_excel('Data_Set2.xlsx')

################################################################
################## Regression modeling #########################
################################################################


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# 1. Define your feature columns and target
# features = ['Region', 'month_sin', 'month_cos', 'Temp', 'sea_level', 'Age_category','age_cat_Monthly_Percent','age_cat_Yearly_Percent',
#             'rolling_3m', 'rolling_6m', 'rolling_12m']
features = ['month_sin', 'month_cos', 'Temp', 'sea_level', 'Age_category','age_cat_Monthly_Percent','age_cat_Yearly_Percent',
            'rolling_3m', 'rolling_6m', 'rolling_12m']

target = 'Number_of_diagnoses'

#Drop rows with NaNs
df_model = combined_all_data.dropna(subset=features + [target, 'Date'])
#Sort by Date
df_model = df_model.sort_values('Date')

# Split based on time (80% train, 20% test)
cutoff = int(len(df_model) * 0.8)
train = df_model.iloc[:cutoff]
test = df_model.iloc[cutoff:]

# Define X and y
X_train = train[features]
X_test = test[features]
y_train = train[target]
y_test = test[target]

# Define column transformer for categorical encoding
# categorical_features = ['Region', 'Age_category']
categorical_features = ['Age_category']
numeric_features = ['month_sin', 'month_cos', 'Temp', 'sea_level','age_cat_Monthly_Percent','age_cat_Yearly_Percent','rolling_3m', 'rolling_6m', 'rolling_12m']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numeric_features)
    ])

# Create pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])


# Fit model
model.fit(X_train, y_train)


# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.5f}")
print(f"R² Score: {r2:.5f}")

################### Anova table ##########################

import statsmodels.api as sm

# Encode categorical features manually (like pipeline does)
X_train_enc = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)
X_test_enc = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)

# Align columns in case one-hot differs between train/test
X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join='left', axis=1, fill_value=0)

# Add constant for intercept
X_train_sm = sm.add_constant(X_train_enc)

# Fit OLS model
model_sm = sm.OLS(y_train, X_train_sm).fit()

# Get summary (includes p-values and ANOVA-style stats)
print(model_sm.summary())
