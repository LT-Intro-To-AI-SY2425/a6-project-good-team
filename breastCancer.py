import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = pd.read_csv("bc/breast_cancer/wdbc_data.csv")
data["Diagnosis"].replace(["1","0"],[1,0],inplace=True)

x = data[["Radius", "Texture", "Smoothness"]].values
y = data["Diagnosis"].values

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)

model = linear_model.LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("accuracy: ", model.score(x_test, y_test))

print(f"Accuracy: {accuracy}")
predictions = model.predict(x_test)
for i in range(len(y_test)): print("prediction: ", predictions[i], " Answer: ", y_test[i])