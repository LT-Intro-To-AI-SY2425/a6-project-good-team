import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("bc/breast_cancer/wdbc_data.csv")
data["Diagnosis"].replace(["M","B"],[0,1],inplace=True)

x = data[["Radius", "Texture", "Smoothness"]].values
y = data["Diagnosis"].values

print(x)
print(y)

x = StandardScaler().fit(x).transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y)

model = linear_model.LogisticRegression()
model.fit(x_train, y_train)

print("accuracy: ", model.score(x_test, y_test))


predictions = model.predict(x_test)
for i in range(len(y_test)): print("prediction: ", predictions[i], " Answer: ", y_test[i])
