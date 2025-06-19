from random import uniform, random, sample, choice
from typing import Callable, List, Type
import numpy as np
import matplotlib.pyplot as plt


def genetic_algorithm(
        model_class: Type,
        f: Callable[[List[float]], float],
        population_size: int,
        generations: int,
        mutation_scale: float,
        crossover_rate: float,
        selection_rate: float = 0.5
):
    population = [model_class(f, [uniform(-10, 10) for _ in range(2)]) for _ in range(population_size)]

    best_trace = []
    population_trace = []

    best_model = None
    best_score = float('inf')

    for gen in range(generations):
        scored = [(m, m.evaluate()) for m in population]
        scored.sort(key=lambda x: x[1])

        current_best = scored[0][0]
        current_score = scored[0][1]

        if current_score < best_score:
            best_model = current_best.copy()
            best_score = current_score

        best_trace.append(current_best.get_state())
        population_trace.append([m.get_state() for m, _ in scored])

        survivors = [m.copy() for m, _ in scored[:int(population_size * selection_rate)]]

        children = []
        while len(children) + len(survivors) < population_size:
            if random() < crossover_rate:
                p1, p2 = sample(survivors, 2)
                child_params = [(x + y) / 2 for x, y in zip(p1.params, p2.params)]
                children.append(model_class(f, child_params))
            else:
                children.append(choice(survivors).copy())

        for m in children:
            m.modify(scale=mutation_scale)

        population = survivors + children

    return best_model, best_score, best_trace, population_trace


def visualize_genetic(f, best_trace, population_trace, every=5):
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-10, 10, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[f([xi, yi]) for xi, yi in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

    plt.figure(figsize=(10, 8))
    contours = plt.contour(X, Y, Z, levels=30, cmap='viridis')
    plt.clabel(contours, inline=True, fontsize=8)

    for i in range(0, len(population_trace), every):
        pop = population_trace[i]
        xs, ys = zip(*pop)
        plt.scatter(xs, ys, c='gray', alpha=0.3, s=15)

    best_xs, best_ys = zip(*best_trace)
    plt.plot(best_xs, best_ys, 'r.-', label='Best path', linewidth=2)

    init_pop = population_trace[0]
    xs0, ys0 = zip(*init_pop)
    plt.scatter(xs0, ys0, c='lime', s=40, label='Initial', edgecolors='black')

    plt.scatter([best_xs[-1]], [best_ys[-1]], c='blue', s=80, label='Best', edgecolors='black')

    plt.title("Genetic Algorithm Trajectory")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
