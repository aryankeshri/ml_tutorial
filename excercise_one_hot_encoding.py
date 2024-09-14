"""
Exercise
At the same level as this notebook on github, there is an Exercise folder that
contains carprices.csv. This file has car sell prices for 3 different models.
First plot data points on a scatter plot chart to see if linear regression model
can be applied. If yes, then build a model that can answer following questions,

1) Predict price of a mercedez benz that is 4 yr old with mileage 45000

2) Predict price of a BMW X5 that is 7 yr old with mileage 86000

3) Tell me the score (accuracy) of your model. (Hint: use LinearRegression().score())
"""
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


df = pd.read_csv('carprices.csv')
print(df.head(3))

# data cleaning operations
dummies = pd.get_dummies(df['Car Model'])
# print(dummies)

df2 = pd.concat([df, dummies], axis='columns')
# print(df2)

df2 = df2.drop(['Car Model', 'Mercedez Benz C class'], axis='columns')
print(df2)

model = linear_model.LinearRegression()
# Training
model.fit(df2[['Mileage', 'Age(yrs)', 'Audi A5', 'BMW X5']], df2[['Sell Price($)']])
# prediction
# print(model.coef_, model.intercept_, model.score(df2[['Mileage', 'Age(yrs)', 'Audi A5', 'BMW X5']], df2[['Sell Price($)']]))

# Predict price of a mercedez benz that is 4 yr old with mileage 45000
print(model.predict([[45000, 4, 0, 0]]))

# Predict price of a BMW X5 that is 7 yr old with mileage 86000
print(model.predict([[86000, 7, 0, 1]]))

