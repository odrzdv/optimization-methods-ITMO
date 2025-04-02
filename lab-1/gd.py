from typing import List, Any, Callable, Tuple, Dict, Union

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


class GradientDescent:
    def __init__(
            self,
            max_iterations: int,
            lr_method_const: float,
            lr_method: str = 'fixed',
            tolerance: float = 1e-5
    ) -> None:
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
        for it in range(self.max_iterations):
            grad: np.ndarray = self.compute_gradient(func, point, with_tracking=True)

            step: float
            prev_f: float = self.history[-1][2]
            match self.lr_method:
                case 'fixed':
                    step = self.lr_method_const
                case 'armijo':
                    step = self.armijo(func, prev_f, point, const=self.lr_method_const)
                case 'exp_decay':
                    step = self.exp_decay(self.lr_method_const, prev_f)
                case 'dec_time':
                    step = self.dec_time(self.lr_method_const, self.history[0][2], it + 1)
                case 'golden_ratio':
                    step = self.golden_ratio(point, func)
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
        parsed_func: Any = sp.parse_expr(func)
        left_dote_x, left_dote_y = X - cur_step * grad[0], Y - cur_step * grad[1]
        left_expr = parsed_func.subs(x, (X - left_dote_x), y, (Y - left_dote_y))
        right_expr = parsed_func.subs(x, X, y, Y) - const * cur_step * (grad[0] ** 2 + grad[1] ** 2)
        if left_expr > right_expr:
            return cur_step * 0.5
        else:
            return cur_step

    def exp_decay(self, decay: float, cur_step: float) -> float:
        return cur_step * decay

    # Тут предпоследнего аргумент всегда значение самого первого шага
    def dec_time(self, gamma: float, init_step: float, number_of_step: int) -> float:
        return init_step / (1 + gamma * number_of_step)

    # Одномерные поиски
    def golden_ratio(self, point: np.ndarray, func: str) -> float:
        a, b = float(point[0]), float(point[1])
        x = sp.symbols("x")
        fi = (np.sqrt(5) + 1) / 2
        c, d = (b - (b - a) / fi), (a + (b - a) / fi)
        parsed_func: Any = sp.parse_expr(func)
        func_c, func_d = parsed_func.subs(x, c), parsed_func.subs(x, d)
        if parsed_func.subs(x, a) <= func_c <= func_d:
            b = d
        else:
            a = c
        return round((a + b) / 2, 5)

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
    gd = GradientDescent(max_iterations=500, lr_method_const=0.1, lr_method='fixed')
    func = "x**2 + y**2"  # f(x, y) = x^2 + y^2
    result = gd.solve(func, init_p=(10, 10))
    print(f"Result: {result['result']}, Iterations: {result['it_cnt']}")
    gd.plot_descent()


main()
