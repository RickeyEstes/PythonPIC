# coding=utf-8
from pythonpic import plotting_parser
from pythonpic.configs.run_laser import initial, impulse_duration, n_macroparticles, number_cells
from pythonpic.visualization.animation import ParticleDensityAnimation

args = plotting_parser("Hydrogen shield")
number_particles = 10000
powers = range(23, 20, -1)
power = 23
intensity = 10**power
# for power in powers:
#     intensity = 10**power
polarizations = ["Ey", "Circular"]
for polarization in polarizations:
    for number_particles, n_cells in [
        [10000, number_cells], #unstable
        # [10000, int(number_cells/2)], #unstable
        # [10000, int(number_cells/3)], #unstable
        # [20000, number_cells], # stable-ish
        # [20000, int(1.5*number_cells)], # stable
        # [40000, int(number_cells*2)], #
        # [75000, int(number_cells)], #
        # [75000, int(number_cells*2)], #
        # [100000, int(number_cells*2)], #
        # [150000, int(number_cells*3)], #
        # [150000, int(number_cells*4)], #
        ]:
        s = initial(f"{number_particles}_{n_cells}_run_{power}_{polarization}", number_particles, n_cells, impulse_duration,
                    intensity, perturbation_amplitude=0, individual_diagnostics=True, laser_polarization=polarization).lazy_run()
        plots(s, *args, frames="few")
        del s