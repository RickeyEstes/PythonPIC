# coding=utf-8
from pythonpic import plotting_parser
from pythonpic.configs.run_laser import laser, impulse_duration, n_macroparticles, plots, number_cells
from pythonpic.visualization.plotting import plots as general_plots
from pythonpic.visualization.animation import ParticleDensityAnimation

args = plotting_parser("Hydrogen shield")
perturbation_amplitude = 0
powers = range(23, 21, -1)
for power in powers:
    intensity = 10**power
    for number_particles, n_cells in [
        [75000, int(number_cells)], #
        # [75000, int(number_cells*2)], #
        ]:
        s = laser(f"{number_particles}_{n_cells}_run_{power}_{perturbation_amplitude}", number_particles, n_cells, impulse_duration,
                  intensity, perturbation_amplitude).lazy_run()
        plots(s, *args, frames="few", animation_type=ParticleDensityAnimation)
        del s
