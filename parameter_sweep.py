# coding=utf-8
from pythonpic import plotting_parser
from pythonpic.configs.run_laser import initial, impulse_duration, n_macroparticles

args = plotting_parser("Hydrogen shield")
perturbation_amplitude = 0

intensity = 1e22
ncells = [500, 1000, 1378, 1500, 1800, 2000]
scaling = 1
for n_cells in ncells:
    s = initial(f"timing_run_ncells_{n_cells}", n_macroparticles, n_cells, impulse_duration, intensity, perturbation_amplitude).lazy_run()
    del s

ncells = 1378
nparticles = [0, 1000, 5000, 10000, 20000, 40000, 60000, 70000, 80000]
for n_particles in nparticles:
    s = initial(f"timing_run_nparticles_{n_particles}", n_particles, ncells, impulse_duration, intensity, perturbation_amplitude).lazy_run()
    del s
