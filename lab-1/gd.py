from typing import List, Any, Callable, Tuple, Dict, Union

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class GradientDescent:
    def __init__(
            self,
            learning_rate: float,
            max_iterations: int,
            lr_method_const: float,
            lr_method: str = 'fixed',
            tolerance: float = 1e-6,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iterations: int = max_iterations
        self.tolerance: float = tolerance
        self.lr_method_const: float = lr_method_const
        self.lr_method = lr_method
        self.history: List[Tuple[float, float, float]] = []
        self.grad_calcs: int = 0
        self.func_calcs: int = 0

    def compute_gradient(self, func: str, point: np.ndarray, with_tracking: bool) -> np.ndarray:
        """
        Computes gradient of func with point

        :param func: function in string representation to compute gradient
        :param point: expected of 2 elements
        :return: gradient of func with point
        :param with_tracking: flag to log results to path history
        """
        variables: List[sp.Symbol] = sp.symbols('x, y')  # defining vars symbols
        parsed_func: Any = sp.parse_expr(func)  # parsing str -> sympy function

        if (with_tracking):
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

        self.grad_calcs = self.grad_calcs + 1
        return np.array([g(point[0], point[1]) for g in grads], dtype=np.float64)  # evaluating gradient at given point

    def solve(self, func: str, init_p: Tuple[float, float]) \
            -> Dict[
                str,
                Union[
                    Tuple[float, float],
                    int
                ]
            ]:
        """
        Computes gradient descent of func with given initial point

        :param func: function of 2 vars in string representation
        :param init_p: initial point of gradient descent (2 vars)
        :return: dict = {'result': result_point, 'it_cnt': count of iterations taken}
        """
        point: np.ndarray = np.array(init_p, dtype=np.float64)  # converting to numpy values
        it_cnt: int = -1
        step: float = self.learning_rate
        for it in range(self.max_iterations):
            grad: np.ndarray = self.compute_gradient(func, point, with_tracking=True)
            # some stuff to check and normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e10 or np.isnan(grad_norm):
                print(f"Warning: Gradient norm is too large or NaN at iteration {it}. Stopping.")
                it_cnt = it
                break
            if grad_norm > 1e5:
                grad = grad / grad_norm * 1e5

            match self.lr_method:
                case 'fixed':
                    step = self.learning_rate
                case 'armijo':
                    step = self.armijo(func, step, point, const=self.lr_method_const)
                case 'exp_decay':
                    step = self.exp_decay(self.lr_method_const)
                case 'dec_time':
                    step = self.dec_time(self.lr_method_const)
                case 'golden_ratio':
                    step = self.golden_ratio(func, point, -grad)
                case 'dichotomy':
                    step = self.dichotomy(point, func, -grad)
                case _:
                    raise ValueError(f'Unknown learning rate method: {self.lr_method}')

            point_k: np.ndarray = point - step * grad
            if np.linalg.norm(point_k - point) < self.tolerance:
                it_cnt = it
                break
            point = point_k
        return {
            'result': (float(point[0]), float(point[1])),
            'it_cnt': it_cnt + 1 if it_cnt >= 0 else self.max_iterations,
            'grad_calcs': self.grad_calcs,
            'func_calcs': self.func_calcs
        }

    def plot_descent(self) -> None:
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
        plt.title('Gradient Descent Path')
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


def objective_function(x):
    return 3 * (x[0] - 3) ** 2 + x[1] ** 2


def main():
    x_sym, y_sym = sp.symbols('x y')
    func = "(1-x)**2 + 100*(y-x**2)**2"
    init_p = (-2, 0)

    # Gradient Descent с suggested params
    gd = GradientDescent(
        learning_rate=0.1,
        max_iterations=1000,
        lr_method_const=0.01,
        lr_method='golden_ratio'
    )
    result_gd = gd.solve(func, init_p)
    print(f"Optimized Gradient Descent Result: {result_gd}")
    gd.plot_descent()


main()
