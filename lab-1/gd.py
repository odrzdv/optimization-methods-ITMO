from typing import List, Any, Callable, Tuple, Dict, Union

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


class GradientDescent:
    def __init__(
            self,
            learning_rate: float,
            max_iterations: int,
            lr_method_const: float,
            lr_method: str = 'fixed',
            tolerance: float = 1e-6
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iterations: int = max_iterations
        self.tolerance: float = tolerance
        self.lr_method_const: float = lr_method_const
        self.lr_method = lr_method
        self.history: List[Tuple[float, float, float]] = []

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
                    step = self.exp_decay(self.lr_method_const, self.learning_rate, it + 1)
                case 'dec_time':
                    step = self.dec_time(self.lr_method_const, self.learning_rate, it + 1)
                case 'golden_ratio':
                    step = self.golden_ratio(func, point, -grad)
                case 'dichotomy':
                    step = self.dichotomy(point, func)
                case _:
                    raise ValueError(f'Unknown learning rate method: {self.lr_method}')

            point_k: np.ndarray = point - step * grad
            if np.linalg.norm(point_k - point) < self.tolerance:
                it_cnt = it
                break
            point = point_k
        return {
            'result': (float(point[0]), float(point[1])),
            'it_cnt': it_cnt + 1 if it_cnt >= 0 else self.max_iterations
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

    # Ток я не понимаю как выбирать const
    def armijo(self, func: str, cur_step: float, point: np.ndarray, const: float) -> float:
        X, Y = float(point[0]), float(point[1])
        x, y = sp.symbols("x y")
        grad = self.compute_gradient(func, point, with_tracking=False)
        parsed_func = sp.parse_expr(func)

        left_dote_x, left_dote_y = X - cur_step * grad[0], Y - cur_step * grad[1]
        left_expr = parsed_func.subs({x: left_dote_x, y: left_dote_y})
        right_expr = parsed_func.subs({x: X, y: Y}) - const * cur_step * (grad[0] ** 2 + grad[1] ** 2)

        if left_expr > right_expr:
            return cur_step * 0.5
        else:
            return cur_step

    def exp_decay(self, const: float, init_lr: float, iteration: int) -> float:
        return init_lr * np.exp(-const * iteration)

    # Тут предпоследнего аргумент всегда значение самого первого шага
    def dec_time(self, gamma: float, init_lr: float, iteration: int) -> float:
        return init_lr / (1 + gamma * iteration)

    # Одномерные поиски
    def golden_ratio(self, func: str, point: np.ndarray, direction: np.ndarray) -> float:
        a, b = 0, 1
        golden_ratio = (np.sqrt(5) - 1) / 2
        epsilon = 1e-5

        x_sym, y_sym = sp.symbols('x y')
        parsed_func = sp.parse_expr(func)
        func_numeric = sp.lambdify((x_sym, y_sym), parsed_func, 'numpy')

        def f_1d(alpha):
            new_point = point + alpha * direction
            return func_numeric(new_point[0], new_point[1])

        alpha1 = a + (1 - golden_ratio) * (b - a)
        alpha2 = a + golden_ratio * (b - a)

        f1 = f_1d(alpha1)
        f2 = f_1d(alpha2)

        while abs(b - a) > epsilon:
            if f1 < f2:
                b = alpha2
                alpha2 = alpha1
                f2 = f1
                alpha1 = a + (1 - golden_ratio) * (b - a)
                f1 = f_1d(alpha1)
            else:
                a = alpha1
                alpha1 = alpha2
                f1 = f2
                alpha2 = a + golden_ratio * (b - a)
                f2 = f_1d(alpha2)

        return (a + b) / 2

    def dichotomy(self, point: np.ndarray, func: str) -> float:
        a, b = float(point[0]), float(point[1])
        x = sp.symbols("x")
        parsed_func: Any = sp.parse_expr(func)
        delta = (b - a) / 4
        m = (a + b) / 2
        func_c, func_d = parsed_func.subs(x, m - delta), parsed_func.subs(x, m + delta)
        if func_c > func_d:
            result = (m - delta + b) / 2
        else:
            result = (a + m + delta) / 2
        return round(result, 5)

    # def compute_step_bold_driver(
    #         self,
    #         increase: float,
    #         decrease: float,
    #         init_step: float,
    #         cur_step: float,
    #         func_decreased: bool,
    # ):
    #     if func_decreased:
    #         return min(init_step, cur_step * increase)
    #     else:
    #         return cur_step * decrease


def main():
    gd = GradientDescent(learning_rate=0.1, max_iterations=500, lr_method_const=0.005, lr_method='golden_ratio')
    func = "x**2 + y**2"  # f(x, y) = x^2 + y^2
    result = gd.solve(func, init_p=(10, 10))
    print(f"Result: {result['result']}, Iterations: {result['it_cnt']}")
    gd.plot_descent()


main()
