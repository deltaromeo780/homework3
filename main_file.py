from dataset_sol import X, y
from functions import (
    set_initial_parameters,
    solve_optimization_task,
    solve_linear_equations_system,
    compare_solutions,
)


def main():
    size = X.shape[1]
    w_init = set_initial_parameters(size)
    w1 = solve_optimization_task(w_init=w_init, X=X, y=y, alpha0=50, epsilon=1e-6)
    w2 = solve_linear_equations_system(X, y)

    compare_solutions(lin_sol=w2, opt_sol=w1)


if __name__ == "__main__":
    main()
