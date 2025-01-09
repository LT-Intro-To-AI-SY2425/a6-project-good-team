import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("")
data['Gender'].replace(['Male','Female'],[0,1],inplace=True)

x = data[["Radius", "Texture", "Smoothness"]].values
y = data["Diagnosis"].values

# Step 1: Print the values for x and y
print(x)
print(y)
# Step 2: Standardize the data using StandardScaler, 
# Step 3: Transform the data
x = StandardScaler().fit(x).transform(x)
# Step 4: Split the data into training and testing data

x_train, x_test, y_train, y_test = train_test_split(x, y)
# Step 5: Fit the data
# Step 6: Create a LogsiticRegression object and fit the data

model = linear_model.LogisticRegression()
model.fit(x_train, y_train)
# Step 7: Print the score to see the accuracy of the model

print("accuracy: ", model.score(x_test, y_test))
# Step 8: Print out the actual ytest values and predicted y values

predictions = model.predict(x_test)
for i in range(len(y_test)): print("prediction: ", predictions[i], " Answer: ", y_test[i])
# based on the xtest data
person = [[34, 56000, 1]]
prediction = model.predict((StandardScaler().fit(x).transform(person)))
print(prediction)