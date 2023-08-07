import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.io as io
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = pd.read_csv("IRIS.csv")
print(iris.head()) 
print()
print(iris.describe())
print("Target Labels", iris["species"].unique())

# Visualize the Iris dataset using Plotly
fig = px.scatter(iris, x="sepal_width", y="sepal_length", color="species")
fig.show()

# Prepare the data for training
x = iris.drop("species", axis=1)
y = iris["species"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Create and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)

# Create a new data point for prediction
new_feature_names = x.columns  # Get the feature names
x_new = pd.DataFrame([[2.9, 6.0, 1.0, 0.2]], columns=new_feature_names)

# Perform prediction on the new data point
prediction = knn.predict(x_new)
print("Prediction: {}".format(prediction))
