# coding=utf-8
from pythonpic.configs.benchmark_run import initial
import numpy as np
import itertools

n_grid = [100, 500, 1000, 2000]
n_particles = [100, 1000, 10000, 50000, 75000]

times_array = np.zeros((len(n_grid), len(n_particles)), dtype=float)

for i, number_particles in enumerate(n_particles):
    for j, number_grid in enumerate(n_grid):
        s = initial(f"{number_particles}_{number_grid}", number_particles,
                    number_grid).lazy_run()
        # s.plots_3d(True, False, True, False)
