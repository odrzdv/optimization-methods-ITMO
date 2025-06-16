from typing import List, Any, Callable, Tuple, Dict, Union
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import solve
import optuna
import warnings


class GradientDescent:
    def __init__(
            self,
            learning_rate: float,
            max_iterations: int,
            lr_method_const: float,
            lr_method: str = 'fixed',
            tolerance: float = 1e-8,
            method: str = 'gradient',  # 'gradient' or 'newton'
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iterations: int = max_iterations
        self.tolerance: float = tolerance
        self.lr_method_const: float = lr_method_const
        self.lr_method = lr_method
        self.method = method  # 'gradient' or 'newton'
        self.history: List[Tuple[float, float, float]] = []
        self.grad_calcs: int = 0
        self.func_calcs: int = 0

    def compute_gradient(self, func: str, point: np.ndarray, with_tracking: bool) -> np.ndarray:
        """
        Computes gradient of func at point.

        :param func: function in string representation to compute gradient
        :param point: expected of 2 elements
        :param with_tracking: flag to log results to path history
        :return: gradient of func at point
        """
        variables: List[sp.Symbol] = sp.symbols('x, y')  # defining vars symbols
        parsed_func: Any = sp.parse_expr(func)  # parsing str -> sympy function

        if with_tracking:
            self.history.append(
                (
                    float(point[0]),
                    float(point[1]),
                    sp.lambdify(variables, parsed_func, modules='numpy')(point[0], point[1])
                )
            )

        part_ders: List[sp.Derivative] = \
            [sp.diff(parsed_func, var) for var in variables]  # calculating partial derivatives

        grads: List[Callable] = []
        for der in part_ders:
            grad_func: Callable = sp.lambdify(variables, der,
                                              modules='numpy')  # converting sympy derivatives to python funcs
            grads.append(grad_func)

        self.grad_calcs += 1
        return np.array([g(point[0], point[1]) for g in grads], dtype=np.float64)  # evaluating gradient at given point

    def compute_hessian(self, func: str, point: np.ndarray) -> np.ndarray:
        """
        Computes Hessian matrix of func at point.

        :param func: function in string representation
        :param point: point where Hessian is computed (2D)
        :return: 2x2 Hessian matrix
        """
        variables: List[sp.Symbol] = sp.symbols('x, y')
        parsed_func: Any = sp.parse_expr(func)

        # Compute second-order partial derivatives
        hessian = np.zeros((2, 2), dtype=np.float64)
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                second_der = sp.diff(sp.diff(parsed_func, var1), var2)
                hessian_func = sp.lambdify(variables, second_der, 'numpy')
                hessian[i, j] = hessian_func(point[0], point[1])

        self.grad_calcs += 2
        return hessian

    def solve_newton_system(self, func: str, point: np.ndarray) -> np.ndarray:
        """
        Solves the Newton system H(x_k) d_k = -∇f(x_k).

        :param func: function in string representation
        :param point: current point x_k
        :return: direction d_k
        """
        gradient = self.compute_gradient(func, point, with_tracking=True)
        hessian = self.compute_hessian(func, point)

        # Solve H(x_k) d_k = -∇f(x_k)
        direction = solve(hessian, -gradient, assume_a='sym')
        return direction

    def solve(self, func: str, init_p: Tuple[float, float]) \
            -> Dict[
                str,
                Union[
                    Tuple[float, float],
                    int
                ]
            ]:
        """
        Computes optimization (gradient descent or Newton's method) of func with given initial point.

        :param func: function of 2 vars in string representation
        :param init_p: initial point of optimization (2 vars)
        :return: dict = {'result': result_point, 'it_cnt': count of iterations taken}
        """
        point: np.ndarray = np.array(init_p, dtype=np.float64)
        it_cnt: int = -1

        x_sym, y_sym = sp.symbols('x y')
        parsed_func_expr = sp.parse_expr(func)
        func_numeric = sp.lambdify((x_sym, y_sym), parsed_func_expr, 'numpy')

        if self.method in ['newton-cg', 'bfgs', 'l-bfgs-b']:
            x_sym, y_sym = sp.symbols('x y')
            parsed_func = sp.parse_expr(func)
            func_numeric = sp.lambdify((x_sym, y_sym), parsed_func, 'numpy')

            def func_scipy(x):
                return func_numeric(x[0], x[1])

            df_dx = sp.diff(parsed_func, x_sym)
            df_dy = sp.diff(parsed_func, y_sym)
            grad_x = sp.lambdify((x_sym, y_sym), df_dx, 'numpy')
            grad_y = sp.lambdify((x_sym, y_sym), df_dy, 'numpy')

            def grad_scipy(x):
                return np.array([grad_x(x[0], x[1]), grad_y(x[0], x[1])])

            def callback(xk):
                f_val = func_scipy(xk)
                self.history.append((xk[0], xk[1], f_val))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(
                    func_scipy,
                    point,
                    method=self.method,
                    jac=grad_scipy,
                    callback=callback,
                    tol=self.tolerance,
                    options={'maxiter': self.max_iterations}
                )

            return {
                'result': (float(res.x[0]), float(res.x[1])),
                'f_value': float(res.fun),
                'it_cnt': res.nit,
                'grad_calcs': res.njev,
                'func_calcs': res.nfev
            }

        for it in range(self.max_iterations):
            if self.method == 'gradient':
                grad: np.ndarray = self.compute_gradient(func, point, with_tracking=True)
                direction = -grad  # Gradient descent direction
            elif self.method == 'newton':
                direction = self.solve_newton_system(func, point)  # Newton direction
            else:
                raise ValueError(f"Unknown method: {self.method}")

            # Normalize direction if too large
            dir_norm = np.linalg.norm(direction)
            if dir_norm > 1e5:
                direction = direction / dir_norm * 1e5

            # Step size selection
            step = self.learning_rate
            match self.lr_method:
                case 'fixed':
                    step = self.learning_rate
                case 'armijo':
                    step = self.armijo(func, step, point, direction, const=self.lr_method_const)
                case 'exp_decay':
                    step = self.exp_decay(it + 1)
                case 'dec_time':
                    step = self.dec_time(it + 1)
                case 'golden_ratio':
                    step = self.golden_ratio(func, point, direction)
                case 'dichotomy':
                    step = self.dichotomy(point, func, direction)
                case _:
                    raise ValueError(f'Unknown learning rate method: {self.lr_method}')

            point_k: np.ndarray = point + step * direction
            if np.linalg.norm(point_k - point) < self.tolerance:
                it_cnt = it
                break
            point = point_k

        return {
            'result': (float(point[0]), float(point[1])),
            'f_value': func_numeric(point[0], point[1]),
            'it_cnt': it_cnt + 1 if it_cnt >= 0 else self.max_iterations,
            'grad_calcs': self.grad_calcs,
            'func_calcs': self.func_calcs
        }

    def plot_descent(self, title) -> None:
        """
        Plots the gradient descent path using the history of points
        """
        history_array = np.array(self.history)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(history_array[:, 0], history_array[:, 1], history_array[:, 2], marker='o', color='r', linestyle='-',
                linewidth=2)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x, y)')
        plt.title(title)
        plt.show()

    def armijo(self, func: str, cur_step: float, point: np.ndarray, direction: np.ndarray, const: float) -> float:
        x_sym, y_sym = sp.symbols('x y')
        parsed_func = sp.parse_expr(func)
        func_numeric = sp.lambdify((x_sym, y_sym), parsed_func, 'numpy')
        grad = self.compute_gradient(func, point, with_tracking=False)
        f_current = func_numeric(point[0], point[1])
        directional_derivative = np.dot(grad, direction)

        # Armijo condition: f(x + αd) ≤ f(x) + c*α*∇f(x)·d
        alpha = cur_step
        max_iter = 20  # Prevent infinite loops
        for _ in range(max_iter):
            new_point = point + alpha * direction
            f_new = func_numeric(new_point[0], new_point[1])
            self.func_calcs += 1
            if f_new <= f_current + const * alpha * directional_derivative:
                return alpha
            alpha *= 0.5

        return alpha

    def exp_decay(self, iteration: int) -> float:
        min_lr = 1e-7
        return max(self.learning_rate * np.exp(-self.lr_method_const * iteration), min_lr)

    def dec_time(self, iteration: int) -> float:
        min_lr = 1e-7
        return max(self.learning_rate / (1 + self.lr_method_const * iteration), min_lr)

    # Одномерные поиски
    def golden_ratio(self, func: str, point: np.ndarray, direction: np.ndarray) -> float:
        a, b = 0, 1
        golden_ratio_val = (np.sqrt(5) - 1) / 2
        epsilon = 1e-5

        x_sym, y_sym = sp.symbols('x y')
        parsed_func = sp.parse_expr(func)
        func_numeric = sp.lambdify((x_sym, y_sym), parsed_func, 'numpy')

        # Tut luchshe tak
        f_1d = lambda alpha: func_numeric(
            point[0] + alpha * direction[0],
            point[1] + alpha * direction[1]
        )

        c = b - golden_ratio_val * (b - a)
        d = a + golden_ratio_val * (b - a)
        f_c = f_1d(c)
        f_d = f_1d(d)
        self.func_calcs += 2

        while abs(b - a) > epsilon:
            if f_c < f_d:
                b = d
                d = c
                f_d = f_c
                c = b - golden_ratio_val * (b - a)
                f_c = f_1d(c)
                self.func_calcs += 1
            else:
                a = c
                c = d
                f_c = f_d
                d = a + golden_ratio_val * (b - a)
                f_d = f_1d(d)
                self.func_calcs += 1

        return (a + b) / 2

    def dichotomy(self, point: np.ndarray, func: str, direction: np.ndarray) -> float:
        a, b = 0, 1  # Init interval che
        epsilon = 1e-5
        delta = epsilon / 4

        x_sym, y_sym = sp.symbols('x y')
        parsed_func = sp.parse_expr(func)
        func_numeric = sp.lambdify((x_sym, y_sym), parsed_func, 'numpy')

        f_1d = lambda alpha: func_numeric(
            point[0] + alpha * direction[0],
            point[1] + alpha * direction[1]
        )

        while (b - a) > epsilon:
            m = (a + b) / 2
            x1 = m - delta
            x2 = m + delta

            f1 = f_1d(x1)
            f2 = f_1d(x2)
            self.func_calcs += 2

            if f1 < f2:
                b = x2
            else:
                a = x1

        return (a + b) / 2


def tune_hyperparameters(method: str, func: str, init_p: Tuple[float, float]) -> dict:
    """
    Tunes hyperparameters using Optuna for gradient descent or Newton method.

    :param method: 'gradient' or 'newton'
    :param func: function to optimize in string form
    :param init_p: initial point
    :return: dictionary with best hyperparameters
    """

    def objective(trial):
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1.0, log=True)
        lr_method = trial.suggest_categorical('lr_method',
                                              ['fixed', 'armijo', 'exp_decay', 'dec_time', 'golden_ratio', 'dichotomy']
                                              )

        lr_const = 0.01
        if lr_method in ['armijo', 'exp_decay', 'dec_time']:
            lr_const = trial.suggest_float('lr_method_const', 1e-5, 1.0, log=True)

        # Create optimizer with suggested hyperparameters
        optimizer = GradientDescent(
            learning_rate=learning_rate,
            max_iterations=1000,
            lr_method_const=lr_const,
            lr_method=lr_method,
            tolerance=1e-8,
            method=method
        )

        result = optimizer.solve(func, init_p)
        return result['f_value']

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    print(f"\nBest hyperparameters for {method}:")
    print(f"  Learning rate: {study.best_params['learning_rate']}")
    print(f"  LR method: {study.best_params['lr_method']}")
    if 'lr_method_const' in study.best_params:
        print(f"  LR constant: {study.best_params['lr_method_const']}")
    print(f"  Best value: {study.best_value}\n")

    return study.best_params


def objective_function(x):
    return 3 * (x[0] - 3) ** 2 + x[1] ** 2


def main():
    x_sym, y_sym = sp.symbols('x y')
    func = "(1-x)**2 + 100*(y-x**2)**2"
    init_p = (-2, 0)
    print("Tuning hyperparameters for Gradient Descent...")
    best_gd_params = tune_hyperparameters('gradient', func, init_p)
    print("Tuning hyperparameters for Newton's Method...")
    best_newton_params = tune_hyperparameters('newton', func, init_p)
    print("\nRunning optimized methods...")

    # Gradient Descent с suggested params
    gd = GradientDescent(
        learning_rate=best_gd_params['learning_rate'],
        max_iterations=1000,
        lr_method_const=best_gd_params.get('lr_method_const', 0.01),
        lr_method=best_gd_params['lr_method'],
        method='gradient'
    )
    result_gd = gd.solve(func, init_p)
    print(f"Optimized Gradient Descent Result: {result_gd}")
    gd.plot_descent('Gradient Descent Path')

    # Newton's Method suggested params
    newton = GradientDescent(
        learning_rate=best_newton_params['learning_rate'],
        max_iterations=1000,
        lr_method_const=best_newton_params.get('lr_method_const', 0.01),
        lr_method=best_newton_params['lr_method'],
        method='newton'
    )
    result_newton = newton.solve(func, init_p)
    print(f"Optimized Newton's Method Result: {result_newton}")
    newton.plot_descent('Newton')

    # Newton-CG Method
    print("\nRunning Newton-CG Method...")
    newton_cg = GradientDescent(
        learning_rate=0.1,
        max_iterations=1000,
        lr_method_const=0.01,
        lr_method='fixed',
        method='newton-cg'
    )
    result_newton_cg = newton_cg.solve(func, init_p)
    print(f"Newton-CG Result: {result_newton_cg}")
    newton_cg.plot_descent('Newton-CG (scipy)')

    # BFGS Method
    print("\nRunning BFGS Method...")
    bfgs = GradientDescent(
        learning_rate=0.1,
        max_iterations=1000,
        lr_method_const=0.01,
        lr_method='fixed',
        method='bfgs'
    )
    result_bfgs = bfgs.solve(func, init_p)
    print(f"BFGS Result: {result_bfgs}")
    bfgs.plot_descent('BFGS (scipy)')

    # L-BFGS-B Method
    print("\nRunning L-BFGS-B Method...")
    lbfgsb = GradientDescent(
        learning_rate=0.1,
        max_iterations=1000,
        lr_method_const=0.01,
        lr_method='fixed',
        method='l-bfgs-b'
    )
    result_lbfgsb = lbfgsb.solve(func, init_p)
    print(f"L-BFGS-B Result: {result_lbfgsb}")
    lbfgsb.plot_descent('L-BFGS-B (scipy)')


main()