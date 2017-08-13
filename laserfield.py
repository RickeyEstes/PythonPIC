# coding=utf-8
"""Laser field propagation only"""
from pythonpic import plotting_parser
from pythonpic.configs.run_laser import initial, impulse_duration, n_macroparticles, plots, number_cells

args = plotting_parser("Hydrogen shield")
perturbation_amplitude = 0
number_particles = 10000
powers = range(23, 20, -1)
power = 23
intensity = 10**power
for number_particles, n_cells in [
    [0, int(number_cells)], #
    ]:
    s = initial(f"{number_particles}_{n_cells}_run_{power}_{perturbation_amplitude}", number_particles, n_cells, impulse_duration,
                intensity, perturbation_amplitude).test_run()
    plots(s, *args, frames="few")
    del s