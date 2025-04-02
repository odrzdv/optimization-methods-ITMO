import typing as tp
from typing import reveal_type

import numpy as np
import sympy as sp
import scipy.optimize as so


class GradientDescent:
    def __init__(
            self,
            learning_rate: float,
            max_iterations: int,
            tolerance: float = 1e-5
    ):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def compute_gradient(self, func: str, x: float, y: float):
        variables: list[sp.Symbol] = sp.symbols('x, y')
        parsed_func: tp.Any = sp.parse_expr(func)
        part_ders: list[sp.Derivative] = [sp.diff(parsed_func, var) for var in variables]
        part_ders_func: list[tp.Any] = [sp.lambdify(variables, g, 'numpy') for g in part_ders]
        return np.array([g(x, y) for g in part_ders_func], dtype=float)

    def solve(self, func: str, init_p: tp.Tuple[float, float]) \
            -> tp.Dict[
                str,
                tp.Union[
                    tp.Tuple[float, float],
                    int
                ]
            ]:
        x, y = init_p
        it_cnt: int = 0
        for it in range(self.max_iterations):
            grad = self.compute_gradient(func, x, y)
            step = self.learning_rate
            x_k, y_k = x - step * grad[0], y - step * grad[1]
            if np.linalg.norm([x_k - x, y_k - y]) < self.tolerance:
                it_cnt = it
                break
            x, y = x_k, y_k
        return {'result': (x, y), 'it_cnt': it_cnt}

    #Ток я не понимаю как выбирать const
    def compute_step_Arhimo(self, func: str, cur_step: float, X: float, Y: float, const: float):
        x, y = sp.symbols("x y")
        grad = self.compute_gradient(func, X, Y)
        parsed_func: tp.Any = sp.parse_expr(func)
        left_dote_x, left_dote_y = X - cur_step * grad[0], Y - cur_step * grad[1]
        left_expr = parsed_func.subs(x, (X - left_dote_x), y, (Y - left_dote_y))
        right_expr = parsed_func.subs(x, X, y, Y) - const * cur_step * (grad[0] ** 2 + grad[1] ** 2)
        if left_expr > right_expr:
            return cur_step * 0.5
        else:
            return cur_step

    def compute_step_exp_decay(self, decay: float, cur_step: float):
        return cur_step * decay

    #Тут последний аргумент всегда значение самого первого шага
    def compute_step_dec_time(self, gamma: float, init_step: float, number_of_step: int):
        return init_step / (1 + gamma * number_of_step)

    def compute_step_bold_driver(
            self,
            increase: float,
            decrease: float,
            init_step: float,
            cur_step: float,
            func_decreased: bool,
    ):
        if func_decreased:
            return min(init_step, cur_step * increase)
        else:
            return cur_step * decrease

    #Одномерные поиски
    def golden_ratio(self, a: float, b: float, func: str):
        x = sp.symbols("x")
        fi = (np.sqrt(5) + 1) / 2
        c, d = (b - (b - a) / fi), (a + (b - a) / fi)
        parsed_func: tp.Any = sp.parse_expr(func)
        func_c, func_d = parsed_func.subs(x, c), parsed_func.subs(x, d)
        if parsed_func.subs(x, a) <= func_c and func_c <= func_d:
            b = d
        else:
            a = c
        return a, b