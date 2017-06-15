""" Run two stream instability"""
# coding=utf-8
import numpy as np

from ..classes import Grid, Simulation, Species
from ..helper_functions import physics

from functools import partial
from ..visualization.plotting import plots
from ..visualization import animation
plots = partial(plots, animation_type = animation.OneDimAnimation)

def stability_condition(k0, v0, w0):
    dimensionless_number = k0 * v0 / w0
    expected_stability = dimensionless_number > 2 ** -0.5
    print(f"k0*v0/w0 is {dimensionless_number} which means the regime is "
          f"{'stable' if expected_stability else 'unstable'}"
          "(stable above sqrt(2))")
    return expected_stability


class two_stream_instability(Simulation):
    def __init__(self, filename,
                           plasma_frequency=1.,
                           qmratio=-1.,
                           T=300 * 0.2,
                           NG=32,
                           N_electrons=128,
                           L=2 * np.pi,
                           epsilon_0=1.0,
                           push_amplitude=0.001,
                           push_mode=1,
                           v0=0.05,
                           vrandom=0.0,
                           species_2_sign=1):
        """Implements two stream instability from Birdsall and Langdon"""
        print("Running two stream instability")
        grid = Grid(T=T, L=L, NG=NG, epsilon_0=epsilon_0)
        print(f"plasma frequency: {plasma_frequency}")
        print(f"timestep: {grid.dt}")
        print(f"iloczyn: {plasma_frequency * grid.dt}")

        physics.check_pusher_stability(plasma_frequency, grid.dt)
        np.random.seed(0)

        particle_mass = 1
        particle_charge = particle_mass * qmratio
        scaling = abs(particle_mass * plasma_frequency ** 2 * L / float(
            particle_charge * N_electrons * epsilon_0))

        physics.check_plasma_parameter(N_electrons * scaling, NG, grid.dx)
        k0 = 2 * np.pi / L

        expected_stability = stability_condition(k0, v0, plasma_frequency)

        electrons1 = Species(particle_charge, particle_mass, N_electrons, grid, "beam1", scaling=scaling)
        electrons2 = Species(species_2_sign * particle_charge, particle_mass, N_electrons, grid, "beam2", scaling=scaling)
        electrons1.v[:, 0] = v0
        electrons2.v[:, 0] = -v0
        list_species = [electrons1, electrons2]
        description = f"Two stream instability - two beams counterstreaming with $v_0$ {v0:.2f}"
        if vrandom:
            description += f" + thermal $v_1$ of standard dev. {vrandom:.2f}"

        description += f" ({'stable' if expected_stability else 'unstable'}).\n"
        self.vrandom = vrandom
        self.push_mode = push_mode
        self.push_amplitude = push_amplitude
        super().__init__(grid, list_species, filename=filename, category_type="twostream", title=description)

    def grid_species_initialization(self):
        for i, species in enumerate(self.list_species):
            species.distribute_uniformly(self.grid.L, 0.5 * self.grid.dx * i)
            species.sinusoidal_position_perturbation(self.push_amplitude, self.push_mode)
            if self.vrandom:
                species.random_velocity_perturbation(0, self.vrandom)


