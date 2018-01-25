# coding=utf-8
from pythonpic import plotting_parser
from pythonpic.configs.run_laser import initial, impulse_duration, n_macroparticles, number_cells

args = plotting_parser("Hydrogen shield")
perturbation_amplitude = 0
powers = [21, 22, 23]
polarizations = ["Ey", "Circular"]
for power, number_particles, n_cells, polarization in [
    [21, 75000, int(number_cells), "Ey"],
    [21, 75000, int(number_cells), "Circular"],
    [23, 75000, int(number_cells), "Ey"],
    [23, 75000, int(number_cells), "Circular"],
]:
    intensity = 10**power
    s = initial(f"{number_particles}_{n_cells}_run_{power}_{polarization}", number_particles, n_cells, impulse_duration,
                intensity, perturbation_amplitude,
                laser_polarization=polarization).lazy_run().plots_3d(*args)
    del s
