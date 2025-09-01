# Fish-weight-prediction

# Step 1 : Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#step 2
# Load Fish dataset
fish = pd.read_csv("https://github.com/ybifoundation/Dataset/raw/main/Fish.csv")

# Preview
fish.head()

fish.info()
fish.describe()

# Check missing values
fish.isnull().sum()

# Visualize weight distribution
sns.histplot(fish["Weight"], bins=30, kde=True)
plt.title("Distribution of Fish Weight")
plt.show()

#Step 3
# Relationship between features and weight
sns.pairplot(fish, x_vars=["Length1","Length2","Length3","Height","Width"], y_vars="Weight", hue="Species", height=3)
plt.show()

#step 4
y = fish["Weight"]

# Include categorical 'Species' as well
X = fish[["Species","Length1","Length2","Length3","Height","Width"]]

#step 5
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=2529)

#step 6
# Encode categorical feature
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(drop="first"), ['Species']),
    ('num', 'passthrough', ['Length1','Length2','Length3','Height','Width'])
])

# Build pipeline
model = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('regressor', LinearRegression())
])

# Train model
model.fit(X_train, y_train)

#step 7
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print("R² Score:", r2)

# Compare Actual vs Predicted
df_compare = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
df_compare.head(10)

#step 8
plt.figure(figsize=(7,7))
plt.scatter(y_test, y_pred, alpha=0.7, color="blue")
plt.plot([0, 1700], [0, 1700], color="red", linestyle="--")
plt.xlabel("Actual Weight")
plt.ylabel("Predicted Weight")
plt.title("Actual vs Predicted Fish Weight")
plt.show()

#step 9
ridge_model = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('regressor', Ridge(alpha=1.0))
])

ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

print("R² Score with Ridge:", r2_score(y_test, y_pred_ridge))


