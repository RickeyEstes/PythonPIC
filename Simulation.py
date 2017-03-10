"""Data interface class"""
# coding=utf-8
import time

import h5py
import numpy as np

from Constants import Constants
from Grid import Grid
from Species import Species
from helper_functions import date_version_string


class Simulation:
    """Contains data from one run of the simulation:
    NT: number of iterations
    NGrid: Number of points on the grid
    NParticle: Number of particles (one species right now)
    L: Length of the simulation domain
    epsilon_0: the physical constant
    particle_positions, velocities: shape (NT, NParticle) numpy arrays of historical particle data
    charge_density, electric_field: shape (NT, NGrid) numpy arrays of historical grid data
    """

    def __init__(self,
                 NT,
                 dt,
                 constants: Constants,
                 grid: Grid,
                 list_species,
                 date_ver_str=date_version_string(),
                 filename=time.strftime("%Y-%m-%d_%H-%M-%S.hdf5"),
                 ):
        """
        :param NT:
        :param dt:
        :param constants:
        :param grid:
        :param list_species:
        """

        self.NT = NT
        self.dt = dt
        self.grid = grid
        self.list_species = list_species
        self.field_energy = np.zeros(NT)
        self.total_energy = np.zeros(NT)
        self.constants = constants
        self.dt = dt
        self.date_ver_str = date_ver_str
        self.filename = filename

    def grid_species_initialization(self):
        """
        Initializes grid and particle relations:
        1. gathers charge from particles to grid
        2. solves Poisson equation to get initial field
        3. initializes pusher via a step back
        """
        self.grid.gather_charge(self.list_species)
        self.grid.solve_poisson()  # TODO: allow for abstract field solver for relativistic case
        # this would go like
        # self.grid.solve_field()
        # and the backend would call solve_poisson or solve_relativistic_bs_poisson_maxwell_whatever
        for species in self.list_species:
            species.init_push(self.grid.electric_field_function, self.dt)

    def iteration(self, i: int):
        """

        :param int i: iteration number
        Runs an iteration step
        1. saves field values
        2. for all particles:
            2. 1. saves particle values
            2. 2. pushes particles forward

        """
        self.grid.save_field_values(i)  # TODO: is this necessary with what happens after loop

        total_kinetic_energy = 0  # accumulate over species
        for species in self.list_species:
            species.save_particle_values(i)
            kinetic_energy = species.push(self.grid.electric_field_function, self.dt).sum()
            species.return_to_bounds(self.grid.L)
            # TODO: remove this sum
            species.kinetic_energy_history[i] = kinetic_energy
            total_kinetic_energy += kinetic_energy

        self.grid.gather_charge(self.list_species)
        fourier_field_energy = self.grid.solve_poisson()
        self.grid.grid_energy_history[i] = fourier_field_energy
        self.total_energy[i] = total_kinetic_energy + fourier_field_energy

    def run(self, save_data: bool = True) -> float:
        """
        Run n iterations of the simulation, saving data as it goes.
        Parameters
        ----------
        save_data (bool): Whether or not to save the data

        Returns
        -------
        runtime (float): runtime of this part of simulation in seconds
        """
        start_time = time.time()
        for i in range(self.NT):
            self.iteration(i)
        runtime = time.time() - start_time

        if save_data:
            self.save_data(filename=self.filename, runtime=runtime)
        return runtime

    def save_data(self, filename: str = time.strftime("%Y-%m-%d_%H-%M-%S.hdf5"), runtime: bool = False) -> str:
        """Save simulation data to hdf5.
        filename by default is the timestamp for the simulation."""

        S = self
        with h5py.File(filename, "w") as f:
            grid_data = f.create_group('grid')
            self.grid.save_to_h5py(grid_data)

            all_species = f.create_group('species')
            for species in S.list_species:
                species_data = all_species.create_group(species.name)
                species.save_to_h5py(species_data)
            f.create_dataset(name="Field energy", dtype=float, data=S.field_energy)
            f.create_dataset(name="Total energy", dtype=float, data=S.total_energy)

            f.attrs['dt'] = S.dt
            f.attrs['NT'] = S.NT
            f.attrs['date_ver_str'] = date_version_string()
            if runtime:
                f.attrs['runtime'] = runtime
        print("Saved file to {}".format(filename))
        return filename

    def __str__(self, *args, **kwargs):
        result_string = f"""
        Simulation from {self.run_date} containing:
        Epsilon zero = {self.constants.epsilon_0}, c = {self.constants.epsilon_0}
        {self.NT} iterations with time step {self.dt}
        {self.grid.NG}-cell grid of length {self.grid.L:.2f}""".lstrip()
        for species in self.list_species:
            result_string = result_string + "\n\t" + str(species)
        return result_string

    def __eq__(self, other: 'Simulation') -> bool:
        result = True
        assert self.date_ver_str == other.date_ver_str, "date not equal!"
        result *= self.date_ver_str == other.date_ver_str

        assert self.constants.epsilon_0 == other.constants.epsilon_0, "epsilon 0 not equal!"
        result *= self.constants.epsilon_0 == other.constants.epsilon_0

        assert self.constants.c == other.constants.c, "epsilon 0 not equal!"
        result *= self.constants.c == other.constants.c

        assert self.NT == other.NT, "NT not equal!"
        result *= self.NT == other.NT

        assert self.dt == other.dt, "NT not equal!"
        result *= self.dt == other.dt

        for this_species, other_species in zip(self.list_species, other.list_species):
            assert this_species == other_species, "{} and {} not equal!".format(this_species.name, other_species.name)
            result *= this_species == other_species
        assert self.grid == other.grid, "grid not equal!"
        result *= self.grid == other.grid
        return result


def load_data(filename: str) -> Simulation:
    """Create a Simulation object from a hdf5 file"""
    with h5py.File(filename, "r") as f:
        total_energy = f['Total energy'][...]

        NT = f.attrs['NT']
        dt = f.attrs['dt']

        grid_data = f['grid']
        NG = grid_data.attrs['NGrid']
        grid = Grid(NT=NT, NG=NG)
        grid.load_from_h5py(grid_data)

        all_species = []
        for species_group_name in f['species']:
            species_group = f['species'][species_group_name]
            species = Species(1, 1, 1, NT=NT)
            species.load_from_h5py(species_group)
            all_species.append(species)
        date_ver_str = f.attrs['date_ver_str']

    S = Simulation(NT, dt, Constants(epsilon_0=grid.epsilon_0, c=1), grid, all_species, date_ver_str)

    S.total_energy = total_energy

    return S
