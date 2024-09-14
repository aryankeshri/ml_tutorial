"""
Download employee retention dataset from here: https://www.kaggle.com/giripujar/hr-analytics.

1. Now do some exploratory data analysis to figure out which variables have direct
    and clear impact on employee retention (i.e. whether they leave the company or continue to work)
2. Plot bar charts showing impact of employee salaries on retention
3. Plot bar charts showing corelation between department and employee retention
4. Now build logistic regression model using variables that were narrowed down in step 1
5. Measure the accuracy of the model
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('HR_comma_sep.csv')
print(df.head(5))

df_left = df[df.left==1]
df_retain = df[df.left==0]
print(df_left.shape, df_retain.shape)

print(df.groupby('left').mean())
"""
From above table we can draw following conclusions,

**Satisfaction Level**: Satisfaction level seems to be relatively low (0.44) in employees leaving the firm vs the retained ones (0.66)
**Average Monthly Hours**: Average monthly hours are higher in employees leaving the firm (199 vs 207)
**Promotion Last 5 Years**: Employees who are given promotion are likely to be retained at firm
"""
# creating the bar plot
pd.crosstab(df.salary, df.left).plot(kind='bar')

# Above bar chart shows employees with high salaries are likely to not leave the company

pd.crosstab(df.Department, df.left).plot(kind='bar')

"""
From the data analysis so far we can conclude that we will use following variables as independant variables in our model
**Satisfaction Level**
**Average Monthly Hours**
**Promotion Last 5 Years**
**Salary**
"""
subdf = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
print(subdf.head())

salary_dummies = pd.get_dummies(subdf.salary, prefix="salary")
df_with_dummies = pd.concat([subdf,salary_dummies],axis='columns')
print(df_with_dummies.head())

df_with_dummies.drop('salary',axis='columns',inplace=True)

X = df_with_dummies
y = df.left

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.3)
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)
print(model.predict(X_test))
print(model.score(X_test,y_test))

# plt.show()
