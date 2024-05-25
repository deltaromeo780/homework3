from sklearn.linear_model import LinearRegression

from dataset_sol import X, y

regressor = LinearRegression().fit(X, y)
# Dopasowujemy model do danych X i y za pomocą metody fit
coefficients = regressor.coef_
price_axis_interception = regressor.intercept_
# regressor.coef_ zwraca współczynniki regresji.
# regressor.intercept_ zwraca punkt przecięcia osi Y.

scikit_sol = coefficients
scikit_sol[0, 0] = price_axis_interception

print(f"scikit solution = {scikit_sol}")
