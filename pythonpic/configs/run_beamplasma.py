""" Run two stream instability"""
# coding=utf-8
import numpy as np

from ..classes import PeriodicGrid, Simulation, Species

from functools import partial
from ..visualization.plotting import plots
from ..visualization import animation, static_plots
plots = partial(plots, animation_type = animation.OneDimAnimation, static_type=static_plots.electrostatic_static_plots)

class weakbeam_instability(Simulation):
    def __init__(self, filename,
                         plasma_frequency=1,
                         qmratio=-1,
                         T=300 * 0.2,
                         NG=32,
                         N_beam=128,
                         N_plasma=2048,
                         L=2 * np.pi,
                         epsilon_0=1,
                         push_amplitude=0.0001,
                         push_mode=1,
                         v0=0.01,
                         vrandom=0,
                         ):
        """Implements beam-plasma instability from Birdsall and Langdon

        * A cold plasma of high density (like in coldplasma)
        * A cold plasma beam of low density injected into the plasma with initial velocity v_0

        Plasma frequency of plasma is much higher than that of beam and it's the dominant one
        wave numbers of interest are near k = plasma's frequency / v_0




        """
        print("Running beam-plasma instability")
        particle_mass = 1
        particle_charge = particle_mass * qmratio

        def scaling(N):
            return abs(particle_mass * plasma_frequency ** 2 * L / float(
                particle_charge * N * epsilon_0))

        grid = PeriodicGrid(L=L, NG=NG, T=T)
        filename = f"data_analysis/BP/{filename}/{filename}.hdf5"

        plasma = Species(particle_charge, particle_mass, N_plasma, grid, "plasma", scaling(N_plasma), individual_diagnostics=True)
        beam = Species(particle_charge, particle_mass, N_beam, grid, "beam2", scaling(N_plasma), individual_diagnostics=True)
        list_species = [beam, plasma]  # , background]
        description = f"Weak beam instability - beam with $v_0$ {v0:.2f} in cold plasma"
        if vrandom:
            description += f" + thermal $v_1$ of standard dev. {vrandom:.2f}"
        description += "\n"
        super().__init__(grid, list_species, filename=filename, category_type="beamplasma", title=description)
        self.v0 = v0
        self.push_amplitude = push_amplitude
        self.push_mode = push_mode
        self.vrandom = vrandom


    def grid_species_initialization(self):
        # total_negative_charge = particle_charge * (N_plasma + N_beam)
        # N_protons = 100
        # q_protons = -total_negative_charge/N_protons
        # proton_mass = 1e16
        beam, plasma = self.list_species
        beam.v[:, 0] = self.v0
        plasma.v[:, 0] = 0
        # background = Species(q_protons, proton_mass, N_protons, "protons", NT, scaling(N_plasma))
        # background.v[:,:] = 0
        for i, species in enumerate(self.list_species):
            species.distribute_uniformly(self.grid.L, 0.5 * self.grid.dx * i)
            species.sinusoidal_position_perturbation(self.push_amplitude, self.push_mode)
            if self.vrandom:
                species.random_velocity_perturbation(0, self.vrandom)
        super().grid_species_initialization()


