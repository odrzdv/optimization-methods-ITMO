import typing as tp
import numpy as np
import sympy as sp

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
