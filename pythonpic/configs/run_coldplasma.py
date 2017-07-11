""" Run cold plasma oscillations"""
# coding=utf-8
from numpy import pi

from ..classes import Grid, Simulation, Species

from functools import partial
from ..visualization.plotting import plots
from ..visualization import animation, static_plots
plots = partial(plots, animation_type = animation.OneDimAnimation, static_type=static_plots.electrostatic_static_plots)

class cold_plasma_oscillations(Simulation):
    def __init__(self, filename,
                             plasma_frequency=1,
                             qmratio=-1,
                             T: float = 150,
                             NG: int = 32,
                             N_electrons: int = 128,
                             L: float = 2 * pi,
                             epsilon_0: float = 1,
                             c: float = 1,
                             push_amplitude: float = 0.001,
                             push_mode: float = 1,
                             **kwargs):
        """
        Runs cold plasma oscillations. Essentially a standing wave.

        Parameters
        ----------
        filename : str
        plasma_frequency : float
            the plasma frequency $\omega_{pe}$ for electrons
        qmratio : float
            The ratio between electron charge and mass. Default is 1.
        T : float
            Duration of the simulation.
        NG : int
            number of grid points
        N_electrons : int
            number of macroparticles
        L : float
            length of the simulation region
        epsilon_0 : float
            the physical constant
        c : float
            speed of light
        push_amplitude : float
            amplitude of initial perturbation
        push_mode : int, float
            wavenumber of initial perturbation
        save_data : bool
        kwargs :

        Returns
        -------
        Simulation
            a `Simulation` object with saved data.
        """
        particle_mass = 1
        particle_charge = particle_mass * qmratio # REFACTOR: use physical units here
        scaling = abs(particle_mass * plasma_frequency ** 2 * L / float(
            particle_charge * N_electrons * epsilon_0))

        grid = Grid(T=T, L=L, NG=NG, epsilon_0=epsilon_0, c=c)

        list_species = [
            Species(N=N_electrons, q=particle_charge, m=particle_mass, grid=grid, name="electrons", scaling=scaling, individual_diagnostics=True),
            ]
        for name, value in kwargs.items():
            if type(value) == Species:
                list_species.append(value)
            print(f"{name}:{value}")

        self.push_mode = push_mode
        self.push_amplitude = push_amplitude

        description = f"Cold plasma oscillations\nposition initial condition perturbed by sinusoidal oscillation mode " \
                      f"{push_mode} excited with amplitude {push_amplitude}\n"

        super().__init__(grid, list_species, filename=filename, category_type="coldplasma", title=description)

    def grid_species_initialization(self):
        for species in self.list_species:
            species.distribute_uniformly(self.grid.L)
            species.sinusoidal_position_perturbation(self.push_amplitude, self.push_mode)
        super().grid_species_initialization()

