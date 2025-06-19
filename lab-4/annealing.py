from abc import ABC, abstractmethod
from random import random, uniform
from typing import Callable, List

import numpy as np
from matplotlib import pyplot as plt


class AnnealingModel(ABC):
    @abstractmethod
    def evaluate(self) -> float:
        pass

    @abstractmethod
    def copy(self) -> 'AnnealingModel':
        pass

    @abstractmethod
    def modify(self, attrs: any) -> None:
        pass

    @abstractmethod
    def rollback(self, previous: 'AnnealingModel') -> None:
        pass

    @abstractmethod
    def get_state(self) -> tuple:
        pass

    def stop(self, iteration: int) -> bool:
        return False


class FunctionModel(AnnealingModel):
    def __init__(self, f: Callable[[List[float]], float], params: List[float]):
        self.f = f
        self.params = params.copy()

    def evaluate(self) -> float:
        if not self.params:
            return float('inf')
        return self.f(self.params)

    def copy(self) -> 'FunctionModel':
        return FunctionModel(self.f, self.params.copy())

    def modify(self, scale: float = 1.0) -> None:
        self.params = [x + uniform(-scale, scale) for x in self.params]

    def rollback(self, previous: 'FunctionModel') -> None:
        self.params = previous.params.copy()

    def get_state(self) -> tuple:
        x = self.params[0] if len(self.params) > 0 else 0.0
        y = self.params[1] if len(self.params) > 1 else 0.0
        return x, y


def simulate_annealing(
        model: AnnealingModel,
        temperature_function: Callable[[int], float],
        get_attrs: Callable[[int], any],
        get_tolerance: Callable[[float, float], float],
        iterations: int
):
    best_model = model.copy()
    best_score = best_model.evaluate()

    trace = [(model.get_state()[0], model.get_state()[1], temperature_function(0))]

    for i in range(1, iterations + 1):
        if model.stop(i):
            break

        T = temperature_function(i)
        attrs = get_attrs(i)

        before = model.evaluate()
        previous = model.copy()

        model.modify(attrs)
        after = model.evaluate()
        delta = after - before

        if delta <= 0:
            if after < best_score:
                best_model = model.copy()
                best_score = after
            trace.append((model.get_state()[0], model.get_state()[1], T))
        else:
            prob = get_tolerance(delta, T)
            print(f"Iter {i}, T={T:.2f}, Δ={delta:.2f}, prob={prob:.2f}, score={after:.2f}")
            if random() > prob:
                model.rollback(previous)
            else:
                trace.append((model.get_state()[0], model.get_state()[1], T))

    return best_model, best_score, trace


def visualize(f, trace):
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-10, 10, 400)
    X, Y = np.meshgrid(x, y)

    Z = np.array([[f([xi, yi]) for xi, yi in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

    xs, ys, ts = zip(*trace)

    plt.figure(figsize=(10, 8))
    contours = plt.contour(X, Y, Z, levels=20, cmap='viridis')
    plt.clabel(contours, inline=True, fontsize=8)

    plt.plot(xs, ys, 'r.-', linewidth=2, markersize=6)

    plt.plot(xs[0], ys[0], 'go', label='Start', markersize=10)
    plt.plot(xs[-1], ys[-1], 'bo', label='End', markersize=10)

    plt.title("Simulated Annealing Trajectory")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
