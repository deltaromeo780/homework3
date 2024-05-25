import numpy as np


def set_initial_parameters(length):
    return (np.random.random(size=(length, 1)) - 0.5) * 10
# funkcja inicjuje początkowe wartości wag
# z przwedziału od 0 do 1, przenosi do -0.5; 0.5 ostateczny przedział -5.5 ; 5.5


def linear_regression_hypothesis(X, w):
    return np.dot(X, w)  # iloczyn skalarny X(m,n) i w(n,1)
# m - liczba próbek n- liczba cech
# Celem funkcji linear_regression_hypothesis jest obliczenie przewidywanych wartości y
# dla danego zbioru cech X i wag w w modelu liniowej regresji.


def error_function(h, y):
    m = len(h)
    errors = h - y  # errors to wektor różnic między przewidywanymi wartościami a rzeczywistymi wartościami.
    return np.dot(errors.T, errors) / (2 * m)


''' np.dot(errors.T, errors) oblicza iloczyn skalarny wektora errors (transponowanego) ze samym sobą, co daje sumę
 kwadratów błędów. Dzielimy tę sumę przez 2 * m, gdzie 2 jest czynnikiem normalizacyjnym dla wygody i zgodności
  z konwencją używaną w regresji liniowej.'''


def error_gradient(h, y, X):
    m = len(y)  # liczba próbek
    gradient = np.dot(X.T, (h - y)) / m
    return gradient


''' h: wektor przewidywanych wartości (hipoteza).
y: wektor rzeczywistych wartości (etykiety).
X: macierz cech, gdzie każdy wiersz reprezentuje jedną próbkę, a każda kolumna reprezentuje jedną cechę.'''


def gradient_descent_step(w, gradient, alpha):
    return w - alpha * gradient  # aktualizacja wag

# w: aktualne wartości wag modelu.
# gradient: gradient funkcji kosztu w punkcie w.
# alpha: współczynnik uczenia, kontrolujący wielkość kroku aktualizacji wag.


def solve_optimization_task(w_init, X, y, alpha0=0.1, epsilon=1e-6):
    # w_init: początkowe wartości współczynników regresji.
    # X: macierz cech.
    # y: wektor wyników.
    # alpha0: początkowy krok uczenia.
    # epsilon: próg zakończenia algorytmu.
    h_init = linear_regression_hypothesis(X, w_init)
    gradient_init = error_gradient(h_init, y, X)

    gradient = gradient_init
    w_prev = w_init
    h_prev = h_init
    alpha = alpha0

    error_prev = error_function(h_prev, y)

    steps = 0

    # h_init oblicza początkową hipotezę regresji liniowej.
    # gradient_init oblicza początkowy gradient błędu.
    # gradient, w_prev, h_prev oraz alpha są inicjalizowane odpowiednio.
    # error_prev oblicza początkową wartość funkcji błędu.
    # steps licznik krokówiteracji.

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

        '''Pętla wykonuje się dopóki norma gradientu jest większa niż epsilon.
        Aktualizuje licznik kroków.
        w jest aktualizowane przez wykonanie kroku gradientu.
        h oblicza nową hipotezę.
        error oblicza nową wartość błędu.
        Jeśli nowy błąd jest większy od poprzedniego, krok uczenia alpha jest zmniejszany.
        Jeśli nowy błąd jest mniejszy lub równy, aktualizowane są h_prev, w_prev, error_prev oraz gradient.
        Jeśli liczba kroków przekracza 5000, algorytm przerywa i informuje, że rozwiązanie nie zostało znalezione.'''

    print(
        f"Optimization task has been ended after {steps} steps, alpha = {alpha}, epsilon = {epsilon}, "
        f"\nObjective function value = {error}"
    )

    return w

    # Po zakończeniu pętli drukuje informacje o liczbie kroków, ostatniej wartości alpha, wartości epsilon oraz
    # końcowej wartości funkcji celu. Funkcja zwraca końcowe współczynniki regresji w.


def solve_linear_equations_system(X, y):
    w = np.dot(np.dot(np.linalg.inv((np.dot(X.T, X))), X.T), y)
    return w


''' Funkcja wykorzystuje wzór analityczny do obliczenia współczynników regresji liniowej 𝑤
 za pomocą pseudoodwrotności macierzy:
𝑤 = (𝑋𝑇 * 𝑋)−1 * 𝑋𝑇𝑦 
np.dot(X.T, X) oblicza iloczyn macierzy transponowanej X i X.
np.linalg.inv(...) oblicza odwrotność powyższej macierzy.
np.dot(..., X.T) oblicza iloczyn odwrotności z macierzą transponowaną X.
np.dot(..., y) oblicza ostateczny iloczyn z wektorem wyników y, dając współczynniki w.
Funkcja zwraca obliczone współczynniki w.'''


def compare_solutions(lin_sol, opt_sol):
    # lin_sol: rozwiązanie uzyskane metodą rozwiązywania układu równań liniowych.
    # opt_sol: rozwiązanie uzyskane metodą optymalizacji.
    print(f"Solution of linear equations system, \nw_lin_sol = \n{lin_sol}")
    print(f"Solution of optimization task, \nw_opt_sol = \n{opt_sol}")

    np.testing.assert_array_almost_equal(lin_sol, opt_sol, decimal=0.001)

# Funkcja używa np.testing.assert_array_almost_equal do porównania, czy oba rozwiązania są prawie równe,
# z dokładnością do 0.001.
# Jeśli rozwiązania różnią się bardziej niż o 0.001, funkcja zgłasza błąd
