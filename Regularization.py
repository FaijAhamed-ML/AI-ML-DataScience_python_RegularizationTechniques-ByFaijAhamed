from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression , Ridge , Lasso # why Ridge? because it's a regularized version of Linear Regression that helps prevent overfitting. Lasso is another regularized version that can also perform feature selection.
from sklearn.metrics import mean_squared_error # For evaluating model performance | why mean squared error? because it gives a good idea of how close the predictions are to the actual values.

# Load the California Housing dataset
california = fetch_california_housing()
X, y = california.data, california.target 

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)
print("  ")
print (" feature names: \n", california.feature_names)
print ("\n sample data: \n", pd.DataFrame(X, columns=california.feature_names).head()) # Display first few rows of the dataset | helps to understand the structure and values of the data.

# Initialize and train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred)
print("\nLinear Regression Mean Squared Error:\n", mse_lr)
print (" ")

# Display model coefficients
print(f"\nLinear Regression Coefficients\n: {lr_model.coef_}")

# train Ridge Regression model
ridge_model = Ridge(alpha=1.0)  # alpha is the regularization strength | higher values mean more regularization
ridge_model.fit(X_train, y_train)

# Make predictions on the test set using Ridge Regression
y_pred_ridge = ridge_model.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
print("\nRidge Regression Mean Squared Error:\n", mse_ridge)
print(" ")

# Display Ridge Regression model coefficients
print(f"\nRidge Regression Coefficients\n: {ridge_model.coef_}")

# train Lasso Regression model
lasso_model = Lasso(alpha=0.1)  # alpha is the regularization strength | higher values mean more regularization
lasso_model.fit(X_train, y_train)

# Make predictions on the test set using Lasso Regression
y_pred_lasso = lasso_model.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
print("\nLasso Regression Mean Squared Error:\n", mse_lasso)
print(" ")

# Display Lasso Regression model coefficients
print(f"\nLasso Regression Coefficients\n: {lasso_model.coef_}")

# Summary of results
print("\nSummary of Mean Squared Errors:")
print(f"Linear Regression: {mse_lr}")
print(f"Ridge Regression: {mse_ridge}")
print(f"Lasso Regression: {mse_lasso}")# The summary helps to compare the performance of different models easily.
# The summary helps to compare the performance of different models easily.

