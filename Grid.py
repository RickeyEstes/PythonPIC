"""The spatial grid"""
# coding=utf-8
import numpy as np
import scipy.fftpack as fft

import algorithms_grid


class Grid:
    """Object representing the grid on which charges and fields are computed
    """

    def __init__(self, L: float = 2 * np.pi, NG: int = 32, epsilon_0: float = 1, NT: float = 1, c: float = 1,
                 dt: float = 1, n_species: int = 1, solver="poisson", bc="sine",
                 bc_params=(1,)):
        """
        :param float L: grid length, in nondimensional units
        :param int NG: number of grid cells
        :param float epsilon_0: the physical constant
        :param int NT: number of timesteps for history tracking purposes
        """
        self.x, self.dx = np.linspace(0, L, NG, retstep=True, endpoint=False)
        self.dt = dt
        self.charge_density = np.zeros(NG + 2)
        self.current_density = np.zeros((NG + 2, 3))
        self.electric_field = np.zeros((NG + 2, 3))
        self.magnetic_field = np.zeros((NG + 2, 2))
        self.energy_per_mode = np.zeros(int(NG / 2))

        self.L = L
        self.NG = int(NG)
        self.NT = NT

        self.c = c
        self.epsilon_0 = epsilon_0
        self.n_species = n_species

        self.charge_density_history = np.zeros((NT, self.NG, n_species))
        self.current_density_history = np.zeros((NT, self.NG, 3, n_species))
        self.electric_field_history = np.zeros((NT, self.NG, 3))
        self.magnetic_field_history = np.zeros((NT, self.NG, 2))

        self.energy_per_mode_history = np.zeros(
            (NT, int(self.NG / 2)))  # OPTIMIZE: get this from efield_history?
        self.grid_energy_history = np.zeros(NT)  # OPTIMIZE: get this from efield_history

        self.solver_string = solver
        self.bc_string = bc

        # specific to Poisson solver but used also elsewhere, for plotting # TODO: clear this part up
        self.k = 2 * np.pi * fft.fftfreq(NG, self.dx)
        self.k[0] = 0.0001
        self.k_plot = self.k[:int(NG / 2)]
        if solver == "direct":
            self.init_solver = self.initial_leapfrog
            self.solve = self.solve_leapfrog
            self.apply_bc = self.leapfrog_bc
            self.previous_field = np.zeros_like(self.electric_field)
        elif solver == "poisson":  # TODO: this should be named Fourier
            self.init_solver = self.initial_poisson
            self.solve = self.solve_poisson
            self.apply_bc = self.poisson_bc
        elif solver == "buneman":
            self.init_solver = self.initial_buneman
            self.solve = self.solve_buneman
            self.apply_bc = self.poisson_bc
        else:
            assert False, "need a solver!"

        self.bc_params = bc_params
        if bc == "sine":
            self.bc_function = algorithms_grid.sine_boundary_condition
        elif bc == "laser":
            self.bc_function = algorithms_grid.laser_boundary_condition

    def direct_energy_calculation(self):
        r"""
        Direct energy calculation as

        :math:`E = \frac{\epsilon_0}{2} \sum_{i=0}^{NG} E^2 \Delta x`

        :return float E: calculated energy
        """
        return self.epsilon_0 * (self.electric_field ** 2).sum() * 0.5 * self.dx

    def solve_poisson(self, neutralize=True):
        r"""
        Solves
        :return float energy:
        """
        self.electric_field[1:-1, 0], self.energy_per_mode = algorithms_grid.PoissonSolver(
            self.charge_density[1:-1], self.k, self.NG, epsilon_0=self.epsilon_0, neutralize=neutralize
            )
        return self.energy_per_mode.sum() / (self.NG / 2)  # * 8 * np.pi * self.k[1]**2

    def initial_poisson(self):
        self.solve_poisson()

    def poisson_bc(self, i):
        pass

    def leapfrog_bc(self, i):
        self.electric_field[0, :] = self.bc_function(i * self.dt, *self.bc_params)

    def initial_buneman(self):
        self.solve_poisson()

    def solve_buneman(self):
        self.electric_field = algorithms_grid.BunemanSolver(self.electric_field, self.current_density, self.dt,
                                                            self.epsilon_0)
        return self.direct_energy_calculation()

    def initial_leapfrog(self):
        self.previous_field[1:-1, 0] = algorithms_grid.LeapfrogWaveInitial(self.electric_field[:, 0],
                                                                           np.zeros_like(self.electric_field[:, 0]),
                                                                           self.c,
                                                                           self.dx,
                                                                           self.dt)

    def solve_leapfrog(self):
        self.electric_field, self.magnetic_field, self.energy_per_mode = algorithms_grid.TransverseWaveSolver(
            self.electric_field, self.magnetic_field, self.current_density, self.dt, self.dx, self.c, self.epsilon_0)
        return self.energy_per_mode

    def gather_charge(self, list_species, i=0):
        self.charge_density[:] = 0.0
        for i_species, species in enumerate(list_species):
            gathered_density = algorithms_grid.charge_density_deposition(self.x, self.dx, species.x[species.alive],
                                                                         species.q)
            self.charge_density_history[i, :, i_species] = gathered_density
            self.charge_density[1:-1] += gathered_density

    def gather_current(self, list_species, i=0):
        self.current_density[:] = 0.0
        for i_species, species in enumerate(list_species):
            gathered_current = algorithms_grid.current_density_deposition(self.x, self.dx, species.x, species.q,
                                                                               species.v)
            self.current_density_history[i, :, :, i_species] = gathered_current
            self.current_density[1:-1] += gathered_current

    def electric_field_function(self, xp):
        # TODO: this only takes x right now
        return algorithms_grid.interpolateField(xp, self.electric_field[1:-1, 0], self.x,
                                                self.dx)  # OPTIMIZE: this is probably slow

    def save_field_values(self, i):
        """Update the i-th set of field values, without those gathered from interpolation (charge\current)"""
        self.electric_field_history[i] = self.electric_field[1:-1]
        self.magnetic_field_history[i] = self.magnetic_field[1:-1]
        self.energy_per_mode_history[i] = self.energy_per_mode
        self.grid_energy_history[i] = self.energy_per_mode.sum() / (self.NG / 2)

    def save_to_h5py(self, grid_data):
        """
        Saves all grid data to h5py file
        grid_data: h5py group in premade hdf5 file
        """

        grid_data.attrs['NGrid'] = self.NG
        grid_data.attrs['L'] = self.L
        grid_data.attrs['epsilon_0'] = self.epsilon_0
        grid_data.create_dataset(name="x", dtype=float, data=self.x)

        grid_data.create_dataset(name="rho", dtype=float, data=self.charge_density_history)
        grid_data.create_dataset(name="current", dtype=float, data=self.current_density_history)
        grid_data.create_dataset(name="Efield", dtype=float, data=self.electric_field_history)
        grid_data.create_dataset(name="Bfield", dtype=float, data=self.magnetic_field_history)

        grid_data.create_dataset(name="energy per mode", dtype=float,
                                 data=self.energy_per_mode_history)  # OPTIMIZE: do these in post production
        grid_data.create_dataset(name="grid energy", dtype=float, data=self.grid_energy_history)

    def load_from_h5py(self, grid_data):
        """
        Loads all grid data from h5py file
        grid_data: h5py group in premade hdf5 file
        """
        self.NG = grid_data.attrs['NGrid']
        self.L = grid_data.attrs['L']
        self.epsilon_0 = grid_data.attrs['epsilon_0']
        self.NT = grid_data['rho'].shape[0]

        # OPTIMIZE: check whether these might not be able to be loaded partially for animation...?
        self.x = grid_data['x'][...]
        self.dx = self.x[1] - self.x[0]
        self.charge_density_history = grid_data['rho'][...]
        self.current_density_history = grid_data['current'][...]
        self.electric_field_history = grid_data['Efield'][...]
        self.magnetic_field_history = grid_data['Bfield'][...]
        self.energy_per_mode_history = grid_data["energy per mode"][
            ...]  # OPTIMIZE: this can be calculated during analysis
        self.grid_energy_history = grid_data["grid energy"][...]

    def __eq__(self, other):
        result = True
        result *= np.isclose(self.x, other.x).all()
        result *= np.isclose(self.charge_density, other.charge_density).all()
        result *= np.isclose(self.electric_field, other.electric_field).all()
        result *= self.dx == other.dx
        result *= self.L == other.L
        result *= self.NG == other.NG
        result *= self.epsilon_0 == other.epsilon_0
        return result


# class RelativisticGrid(Grid):
#     def __init__(self, L=2 * np.pi, NG=32, epsilon_0=1, c=1, NT=1):
#         super().__init__(L, NG, epsilon_0, NT)
#         self.c = c
#         self.dt = self.dx / c
#         self.Jyplus = np.zeros_like(self.x)
#         self.Jyminus = np.zeros_like(self.x)
#         self.Fplus = np.zeros_like(self.x)
#         self.Fminus = np.zeros_like(self.x)
#         self.Ey_history = np.zeros((NT, self.NG))
#         self.Bz_history = np.zeros((NT, self.NG))
#         self.current_density_history = np.zeros((NT, self.NG, 3))
#
#     # TODO: implement LPIC-style field solver
#
#     def iterate_EM_field(self):
#         """
#         calculate Fplus, Fminus in next iteration based on their previous
#         values
#
#         assumes fixed left ([0]) boundary condition
#
#         F_plus(n+1, j) = F_plus(n, j) - 0.25 * dt * (Jyminus(n, j-1) + Jplus(n, j))
#         F_minus(n+1, j) = F_minus(n, j) - 0.25 * dt * (Jyminus(n, j+1) - Jplus(n, j))
#
#         TODO: check viability of laser BC
#         take average of last term instead at last point instead
#         """
#
#         # TODO: get laser boundary condition from Birdsall
#         self.Fminus[-1] = self.Fminus[-2] - 0.25 * self.dt * (self.Jyplus[0] - self.Jyminus[-1])
#
#     def unroll_EyBz(self):
#         return self.Fplus + self.Fminus, self.Fplus - self.Fminus
#
#     def apply_laser_BC(self, B0, E0):
#         self.Fplus[0] = (E0 + B0) / 2
#         self.Fminus[0] = (E0 - B0) / 2


if __name__ == "__main__":
    pass
