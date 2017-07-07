# coding=utf-8
from pythonpic import plotting_parser
from pythonpic.configs.run_laser import laser, impulse_duration, n_macroparticles, plots, number_cells
from pythonpic.visualization.plotting import plots as general_plots
from pythonpic.visualization.animation import ParticleDensityAnimation, FastAnimation, FullAnimation

args = plotting_parser("Hydrogen shield")
perturbation_amplitude = 0
powers = [21, 22, 23]
polarizations = ["Ey", "Circular"]
for polarization in polarizations:
    for power in powers:
        intensity = 10**power
        for number_particles, n_cells in [
            [75000, int(number_cells)],
            ]:
            s = laser(f"{number_particles}_{n_cells}_run_{power}_{polarization}", number_particles, n_cells, impulse_duration,
                      intensity, perturbation_amplitude, laser_polarization=polarization).lazy_run()
            plots(s, *args, frames="few", animation_type=FullAnimation)
            del s
