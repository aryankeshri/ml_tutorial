import math

from word2number import w2n
import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv('hiring.csv')
print(df)

df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(
    math.floor(df['test_score(out of 10)'].mean()))
df.experience = df.experience.fillna('zero')
print(df)
df.experience = df.experience.apply(w2n.word_to_num)
print(df)

reg = linear_model.LinearRegression()
reg.fit(df[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']].values,
        df['salary($)'])

predict_salary = reg.predict([[2, 9, 6]])
print(predict_salary)
print(reg.coef_)
print(reg.intercept_)
