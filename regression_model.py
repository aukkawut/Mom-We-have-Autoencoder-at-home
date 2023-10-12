from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import r2_score
# use wine data
data = datasets.load_wine()
X = data.data
y = data.target

# Train the model with lasso regression, find significant predictors
lasso = linear_model.Lasso(alpha=0.1)
lasso.fit(X, y)
significant_features = np.where(lasso.coef_ != 0)[0]

# Use only the significant predictors to fit a standard linear regression model
X_significant = data.data[:, significant_features]
linear_reg = linear_model.LinearRegression()
linear_reg.fit(X_significant, y)

# Get the coefficients and intercept of the linear regression model
coefficients = linear_reg.coef_
intercept = linear_reg.intercept_

print(significant_features, coefficients, intercept)

def adjusted_r2(r2, n, k):
    return 1 - ((1 - r2) * (n - 1) / (n - k - 1))

# 1. Predict with the model with the subset of predictors and compute R^2
y_pred_subset = linear_reg.predict(X_significant)
r2_subset = r2_score(y, y_pred_subset)

# 2. Predict with the model with all predictors (using Lasso) and compute R^2
y_pred_all = lasso.predict(X)
r2_all = r2_score(y, y_pred_all)

n = len(y)
k_subset = len(significant_features)
k_all = X.shape[1]

# Calculate adjusted R^2 for both models again
adj_r2_subset = adjusted_r2(r2_subset, n, k_subset)
adj_r2_all = adjusted_r2(r2_all, n, k_all)

print(adj_r2_subset, adj_r2_all)

# TODO: - Generate data using the pruned model
#       - Train the model with the pruned data on other non-linear model, compare the results with full data