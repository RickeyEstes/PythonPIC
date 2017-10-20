# coding=utf-8
from pythonpic import plotting_parser
from pythonpic.configs.run_laser import initial, impulse_duration, n_macroparticles, plots

args = plotting_parser("Hydrogen shield")
perturbation_amplitude = 0
intensities = [5e20, 1e21, 5e21, 1e21, 5e21, 1e22, 5e22, 1e23]
scalings = [0.5, 0.9, 1, 1.1, 1.5]
for intensity in intensities:
    for scaling in scalings:
        s = initial(f"production_run_{intensity}_{scaling}", n_macroparticles, impulse_duration, intensity, perturbation_amplitude, scaling).lazy_run()
        if any(args):
            plots(s, *args, frames="few")
        del s
