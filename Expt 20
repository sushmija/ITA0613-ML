import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.io as io
io.renderers.default = 'browser'

# Load the data
data = pd.read_csv("futuresale prediction.csv")
print(data.head())
print(data.sample(5))
print(data.isnull().sum())

# Visualize the relationships between features and the target variable (Sales)
import plotly.express as px
import plotly.graph_objects as go

figure = px.scatter(data_frame=data, x="Sales", y="TV", size="TV", trendline="ols")
figure.show()

figure = px.scatter(data_frame=data, x="Sales", y="Newspaper", size="Newspaper", trendline="ols")
figure.show()

figure = px.scatter(data_frame=data, x="Sales", y="Radio", size="Radio", trendline="ols")
figure.show()

# Check the correlation between features and the target variable (Sales)
correlation = data.drop("Sales", axis=1).corrwith(data["Sales"]).sort_values(ascending=False)
print(correlation)

# Prepare the data for training and testing
x = data.drop("Sales", axis=1).values
y = data["Sales"].values
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(xtrain, ytrain)

# Evaluate the model on the test data
print("R-squared score on test data:", model.score(xtest, ytest))

# Make predictions using the model
features = np.array([[230.1, 37.8, 69.2]])  # Example feature values for TV, Radio, and Newspaper
prediction = model.predict(features)
print("Predicted Sales:", prediction)
