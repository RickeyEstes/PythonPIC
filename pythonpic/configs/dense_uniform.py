# coding=utf-8
"""Implements interaction of the laser with a hydrogen shield plasma"""
from functools import partial

import numpy as np

from pythonpic.classes import Grid, Simulation, Species
from pythonpic.helper_functions.physics import epsilon_zero, electric_charge, lightspeed, proton_mass, \
    electron_rest_mass, \
    critical_density, cold_plasma_frequency
from pythonpic.visualization import animation
from pythonpic.visualization.plotting import plots

plots = partial(plots, animation_type = animation.FullAnimation, alpha=0.3)

VERSION = 23
laser_wavelength = 1.064e-6 # meters
laser_intensity = 1e23 # watt/meters squared
impulse_duration = 1e-13 # seconds

length = 1.0655e-5 # meters
total_time = 2e-13 # seconds
spatial_step = 7.7325e-9 # meters
number_cells = 1378

moat_length_left_side = 3.093e-6 # meters
# linear preplasma
preplasma_length = 7.73e-7 # meters
main_plasma_length = 7.73e-7 + preplasma_length # meters

print("crit density", critical_density(laser_wavelength))
maximum_electron_concentration = 5 * critical_density(laser_wavelength) # m^-3

# assert np.isclose(maximum_electron_concentration, 5.24e27), maximum_electron_concentration # m^-3
# maximum_electron_concentration = 5.24e27 # CHECK: this is a crutch

npic = 0.01 * critical_density(laser_wavelength)


scaling = npic# CHECK what should be the proper value here?

category_name = "stability"
# assert False
class uniform(Simulation):
    def __init__(self, filename, n_macroparticles, n_cells):
        """
        A simulation of laser-hydrogen shield interaction.

        Parameters
        ----------
        filename : str
            Filename for the simulation.
        n_macroparticles : int
            Number of macroparticles for each species. The simulation is
            normalized to 75000 macroparticles by default,
        n_cells : int
            Number of grid cells.
        """
        grid = Grid(T=total_time, L=length, NG=int(n_cells), c =lightspeed, epsilon_0 =epsilon_zero, periodic=True)


        cells_per_wl = laser_wavelength / grid.dx
        print(cells_per_wl)
        vtherm = 2 * np.pi / cells_per_wl * lightspeed * 10
        print(vtherm / lightspeed)


        if n_macroparticles:
            electrons = Species(-electric_charge, electron_rest_mass, n_macroparticles, grid, "electrons", scaling)
            electrons.random_velocity_init(vtherm)
            protons = Species(electric_charge, proton_mass, n_macroparticles, grid, "protons", scaling)
            list_species = [electrons, protons]

            omega_p = (cold_plasma_frequency(electrons.scaling * electrons.N / grid.dx, electron_rest_mass, epsilon_zero, electric_charge)**2 + 3 * (grid.NG * 2 * np.pi / grid.L * vtherm)**2)**0.5
            debye_length = vtherm / omega_p
            print(grid.dx, debye_length)
            print("grid stability", grid.dx , 3.4 * debye_length, grid.dx < 3.4 * debye_length)
            print("step stability", grid.dt , 2/omega_p, grid.dt < 2/omega_p)
        else:
            list_species = []

        description = "Stability test"

        super().__init__(grid, list_species,
                         filename=filename,
                         category_type="stability",
                         config_version=VERSION,
                         title=description,
                         considered_large = True)
        print("Simulation prepared.")

    def grid_species_initialization(self):
        for species in self.list_species:
            print(f"Distributing {species.name} uniformly.")
            species.distribute_uniformly(self.grid.L)
        print("Finished initial distribution of particles.")
        super().grid_species_initialization()
        print("Finished initialization.")


