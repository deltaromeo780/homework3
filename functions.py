import numpy as np


def set_initial_parameters(length):
    return (np.random.random(size=(length, 1)) - 0.5) * 10


def linear_regression_hypothesis(X, w):
    return np.dot(X, w)


def error_function(h, y):
    m = len(h)
    errors = h - y
    return np.dot(errors.T, errors) / (2 * m)


def error_gradient(h, y, X):
    m = len(y)
    gradient = np.dot(X.T, (h - y)) / m
    return gradient


def gradient_descent_step(w, gradient, alpha):
    return w - alpha * gradient


def solve_optimization_task(w_init, X, y, alpha0=0.1, epsilon=1e-6):
    h_init = linear_regression_hypothesis(X, w_init)
    gradient_init = error_gradient(h_init, y, X)

    gradient = gradient_init
    w_prev = w_init
    h_prev = h_init
    alpha = alpha0

    error_prev = error_function(h_prev, y)

    steps = 0

    while np.sqrt(np.dot(gradient.T, gradient)) > epsilon:
        steps += 1
        w = gradient_descent_step(w_prev, gradient, alpha)
        h = linear_regression_hypothesis(X, w)
        error = error_function(h, y)
        if error > error_prev:
            alpha = alpha / 2
        else:
            h_prev = h
            w_prev = w
            error_prev = error
            gradient = error_gradient(h, y, X)

        if steps > 5000:
            print("Optimal solution hasn't been found")
            break

    print(
        f"Optimization task has been ended after {steps} steps, alpha = {alpha}, epsilon = {epsilon}, "
        f"\nObjective function value = {error}"
    )

    return w


def solve_linear_equations_system(X, y):
    w = np.dot(np.dot(np.linalg.inv((np.dot(X.T, X))), X.T), y)
    return w


def compare_solutions(lin_sol, opt_sol):
    print(f"Solution of linear equations system, \nw_lin_sol = \n{lin_sol}")
    print(f"Solution of optimization task, \nw_opt_sol = \n{opt_sol}")

    np.testing.assert_array_almost_equal(lin_sol, opt_sol, decimal=0.001)
