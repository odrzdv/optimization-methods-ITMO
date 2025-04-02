from typing import List, Any, Callable, Tuple, Dict, Union

import numpy as np
import sympy as sp

class GradientDescent:
    def __init__(
            self,
            learning_rate: float,
            max_iterations: int,
            tolerance: float = 1e-5
    ) -> None:
        self.learning_rate: float = learning_rate
        self.max_iterations: int = max_iterations
        self.tolerance: float = tolerance

    def compute_gradient(self, func: str, point: np.ndarray) -> np.ndarray:
        """
        Computes gradient of func with point

        :param func: function in string representation to compute gradient
        :param point: expected of 2 elements
        :return: gradient of func with point
        """
        variables: List[sp.Symbol] = sp.symbols('x, y') # defining vars symbols
        parsed_func: Any = sp.parse_expr(func) # parsing str -> sympy function
        part_ders: List[sp.Derivative] = \
            [sp.diff(parsed_func, var) for var in variables] # calculating partial derivatives

        grads: List[Callable] = []
        for der in part_ders:
            grad_func: Callable = sp.lambdify(variables, der, modules='numpy') # converting sympy derivatives to python funcs
            grads.append(grad_func)

        return np.array([g(point[0], point[1]) for g in grads], dtype=np.float64) # evaluating gradient at given point

    def solve(self, func: str, init_p: Tuple[float, float]) \
            -> Dict[
                str,
                Union[
                    Tuple[float, float],
                    int
                ]
            ]:
        """

        :param func: function of 2 vars in string representation
        :param init_p: initial point of gradient descent (2 vars)
        :return: dict = {'result': result_point, 'it_cnt': count of iterations taken}
        """
        point: np.ndarray = np.array(init_p, dtype=np.float64) # converting to numpy values
        it_cnt: int = -1
        for it in range(self.max_iterations):
            grad: np.ndarray = self.compute_gradient(func, point)
            step: float = self.learning_rate
            point_k: np.ndarray = point - step * grad
            if np.linalg.norm(point_k - point) < self.tolerance:
                it_cnt = it
                break
            point = point_k
        return {
            'result': (float(point[0]), float(point[1])),
            'it_cnt': it_cnt + 1 if it_cnt >= 0 else self.max_iterations
        }
