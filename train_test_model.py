import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv('carprices_train_test.csv')
print(df)

x = df[['Mileage' , 'Age(yrs)']].values
y = df[['Sell Price($)']].values
print(x, y)

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.33)
print(X_train, X_test, y_train, y_test)

clf = LinearRegression()
clf.fit(X_train, y_train)

print(clf.predict([[87600, 8]]))
print(clf.score(X_train, y_train))