from sklearn.linear_model import LinearRegression

from dataset_sol import X, y

regressor = LinearRegression().fit(X, y)
coefficients = regressor.coef_
price_axis_interception = regressor.intercept_

scikit_sol = coefficients
scikit_sol[0, 0] = price_axis_interception

print(f"scikit solution = {scikit_sol}")
