import math
import random
from functools import partial
from typing import List

from annealing import FunctionModel, simulate_annealing, visualize, TSPModel, plot_tsp_route
from genetic import genetic_algorithm, visualize_genetic


def make_temperature_function(a=5, b=0.001):
    return lambda i: a * math.exp(-b * i)


def zero_temperature(i):
    return 1e-8


def inv_square_attrs(i: int) -> float:
    # return 2 / math.sqrt(i)
    return max(0.7, 2 / math.sqrt(i))


def fixed_attrs(i: int) -> float:
    return 0.7


def get_tolerance(delta, t):
    return math.exp(-delta / t)


def ellipse(params: List[float]) -> float:
    x = params[0]
    y = params[1]
    return x ** 2 + 3 * y ** 2


def noisy_ellipse(params: List[float], m: float = 5.0) -> float:
    x, y = params[0], params[1]
    base = x ** 2 + 3 * y ** 2
    noise = math.sin(5 * x) * math.sin(3 * y) * math.sin(math.pi * x) * math.sin(math.pi * y / 2) * m
    return base + noise


def multimodal_function(params: List[float], m: float = 5.0) -> float:
    x, y = params[0], params[1]
    base = 0.1 * x ** 2 + 0.1 * y ** 2
    wells = math.sin(x - math.pi / 2) + math.sin(y - math.pi / 2)
    return base + m * wells


noisy_func = partial(noisy_ellipse, m=3)
multimodal = partial(multimodal_function, m=5)
#
# model = FunctionModel(
#     # f=lambda params: sum(noisy_func(params) for _ in range(5)) / 5,
#     f=multimodal,
#     # f=ellipse,
#     params=[random.uniform(-10, 10) for _ in range(2)]
# )
#
# result, score, trace = simulate_annealing(
#     model=model,
#     temperature_function=make_temperature_function(a=100, b=0.01),
#     # temperature_function=zero_temperature,
#     get_attrs=inv_square_attrs,
#     # get_attrs=fixed_attrs,
#     get_tolerance=get_tolerance,
#     iterations=500
# )
#
# print(f"Best x = {result.get_state()[0]:.5f}, y = {result.get_state()[1]:.5f}, f(x, y) = {score:.5f}")
#
#
# def downsample_trace(trace, step=5):
#     return trace[::step] + [trace[-1]]
#
#
# visualize(
#     # ellipse,
#     # noisy_func,
#     multimodal,
#     downsample_trace(trace, step=10)
# )
#
# best_model, score, best_trace, pop_trace = genetic_algorithm(
#     model_class=FunctionModel,
#     # f=multimodal,
#     # f=ellipse,
#     f=noisy_func,
#     population_size=70,
#     generations=200,
#     mutation_scale=0.4,
#     crossover_rate=0.8,
#     selection_rate=0.5
# )
#
# print(f"Best: x={best_model.get_state()[0]:.3f}, y={best_model.get_state()[1]:.3f}, score={score:.5f}")
#
# visualize_genetic(
#     # ellipse,
#     noisy_func,
#     # multimodal,
#     best_trace,
#     pop_trace,
#     every=5
# )


cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(30)]
model = TSPModel(cities)

result, cost, trace = simulate_annealing(
    model=model,
    temperature_function=make_temperature_function(a=1000, b=0.005),
    get_attrs=lambda i: None,
    get_tolerance=lambda delta, T: math.exp(-delta / T),
    iterations=10000
)

plot_tsp_route(cities, result.route, title=f"TSP Simulated Annealing (cost={cost:.2f})")
