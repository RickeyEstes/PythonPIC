"""The spatial grid"""
# coding=utf-8
import numpy as np
import h5py
import scipy.fftpack as fft

from ..helper_functions import physics
from ..algorithms import charge_deposition, FieldSolver, BoundaryCondition, current_deposition, field_interpolation


class Grid:
    """
    Object representing the grid on which charges and fields are computed
    """


    def __init__(self, T: float, L: float, NG: int, c: float = 1, epsilon_0: float = 1, bc=lambda *x: None,
                 periodic=True):
        """

        Parameters
        ----------
        T : float
            total runtime of the simulation
        L : float
            total length of simulation area
        NG : int
            number of grid cells
        c : float
            speed of light
        epsilon_0 : float
            electric permittivity of vacuum
        bc : function
            Function for providing values of the left boundary. To be refactored into taking the Laser object. # REFACTOR
        periodic: bool
            Defines whether the grid is to be treated as periodic or non-periodic.
        """


        self.c = c
        self.epsilon_0 = epsilon_0
        self.particle_bc = lambda *x: None
        self.x, self.dx = np.linspace(0, L, NG, retstep=True, endpoint=False, dtype=np.float64)
        self.x_interpolation = np.arange(NG+2)*self.dx - self.dx

        self.dt = self.dx / c
        self.T = T
        self.NT = physics.calculate_number_timesteps(T, self.dt)
        self.epsilon_0 = epsilon_0

        self.charge_density = np.zeros(NG + 1, dtype=np.float64)
        self.current_density_x = np.zeros((NG + 3), dtype=np.float64)
        self.current_density_yz = np.zeros((NG + 4, 2), dtype=np.float64)
        self.electric_field = np.zeros((NG + 2, 3), dtype=np.float64)
        self.magnetic_field = np.zeros((NG + 2, 3), dtype=np.float64)

        self.L = L
        self.NG = NG

        self.bc_function = bc # REFACTOR boundary condition
        self.k = 2 * np.pi * fft.fftfreq(self.NG, self.dx)
        self.k[0] = 0.0001

        self.periodic = periodic
        if self.periodic:
            self.charge_gather_function = charge_deposition.periodic_density_deposition
            self.current_longitudinal_gather_function = current_deposition \
                .periodic_longitudinal_current_deposition
            self.current_transversal_gather_function = current_deposition.periodic_transversal_current_deposition
            self.particle_bc = BoundaryCondition.return_particles_to_bounds
            self.interpolator = field_interpolation.PeriodicInterpolateField
            self.solver = FieldSolver.BunemanSolver
        else:
            self.charge_gather_function = charge_deposition.density_deposition
            self.current_longitudinal_gather_function = current_deposition.aperiodic_longitudinal_current_deposition
            self.current_transversal_gather_function = current_deposition.aperiodic_transversal_current_deposition
            self.particle_bc = BoundaryCondition.kill_particles_outside_bounds
            self.interpolator = field_interpolation.AperiodicInterpolateField
            self.solver = FieldSolver.BunemanSolver

        self.list_species = []
        self.postprocessed = False
        self.postprocessed_fourier = False

    def prepare_history_arrays_h5py(self, f):
        self.file = f
        group = self.file.create_group("grid")
        self.charge_density_history = group.create_dataset(name="rho", dtype=float, shape=(self.NT, self.NG))
        self.current_density_history = group.create_dataset(name="current", dtype=float, shape=(self.NT, self.NG, 3))
        self.electric_field_history = group.create_dataset(name="Efield", dtype=float, shape=(self.NT, self.NG, 3))
        self.magnetic_field_history = group.create_dataset(name="Bfield", dtype=float, shape=(self.NT, self.NG, 3))
        self.laser_energy_history = group.create_dataset(name="laser", dtype=float, shape=(self.NT,))
        group.create_dataset(name="x", dtype=float, data=self.x)

        h5py_dictionary = {'NGrid': self.NG,
                           'L': self.L,
                           'epsilon_0': self.epsilon_0,
                           'c': self.c,
                           'dt': self.dt,
                           'dx': self.dx,
                           'NT': self.NT,
                           'T': self.T,
                           'periodic': self.periodic,
                           'postprocessed': self.postprocessed,
                           'postprocessed_fourier': self.postprocessed_fourier
                           }
        for key, value in h5py_dictionary.items():
            group.attrs[key] = value


    def postprocess_fourier(self):
        group = self.file['grid']
        if not group.attrs["postprocessed_fourier"]:
            self.longitudinal_energy_history  = group.create_dataset("longitudinal_energy", data=0.5 * self.epsilon_0 * (self.electric_field_history[:,:,0] ** 2))
            perpendicular_electric_energy = 0.5 * self.epsilon_0 * (self.electric_field_history[:,:,1:] ** 2).sum(2) # over directions
            mu_zero_inv = 1/ (self.epsilon_0 * self.c**2)
            magnetic_energy = 0.5 * (self.magnetic_field_history[...] **2).sum(2) * mu_zero_inv # over directions

            self.perpendicular_energy_history = group.create_dataset("perpendicular_energy", data=perpendicular_electric_energy + magnetic_energy)
            self.check_on_charge = group.create_dataset("charge_test", data=np.gradient(self.electric_field_history[:, :, 0], self.dx, axis=1) * self.epsilon_0)
            # fourier analysis
            from scipy import fftpack
            self.k_plot = group.create_dataset("k_plot", data=fftpack.rfftfreq(int(self.NG), self.dx)[::2])
            self.longitudinal_energy_per_mode_history = group.create_dataset("longitudinal_fourier", data=np.abs(fftpack.rfft(self.longitudinal_energy_history))[:,::2])
            self.perpendicular_energy_per_mode_history = group.create_dataset("perpendicular_fourier", data=np.abs(fftpack.rfft(self.perpendicular_energy_history))[:,::2])

            self.longitudinal_energy_history  = group.create_dataset("total_longitudinal", data=self.longitudinal_energy_history[...].sum(1))
            self.perpendicular_energy_history = group.create_dataset("total_perpendicular", data=self.perpendicular_energy_history[...].sum(1))
            self.grid_energy_history = group.create_dataset("total_grid", data=self.perpendicular_energy_history[...] + self.longitudinal_energy_history[...])
            group.attrs["postprocessed_fourier"] = True
            self.postprocessed_fourier = True
            self.file.flush()
        else:
            self.perpendicular_energy_history = group["perpendicular_energy"]
            self.check_on_charge = group["charge_test"]
            self.k_plot = group["k_plot"]
            self.longitudinal_energy_per_mode_history = group["longitudinal_fourier"]
            self.perpendicular_energy_history = group["perpendicular_fourier"]
            self.longitudinal_energy_history = group["total_longitudinal"]
            self.perpendicular_energy_history = group["total_perpendicular"]
            self.grid_energy_history = group["total_grid"]

    def postprocess(self, fourier=True):
        group = self.file['grid']
        if not self.postprocessed:
            print("Postprocessing grid.")
            self.t = group.create_dataset(name="t", data=np.arange(self.NT) * self.dt)
            if fourier:
                self.postprocess_fourier()
            vacuum_wave_impedance= 1/ (self.epsilon_0 * self.c)
            self.laser_energy_history[...] = np.cumsum(self.laser_energy_history[...]**2/ vacuum_wave_impedance * self.dt)
            self.x_current = group.create_dataset(name="x_current", data=self.x + self.dx / 2)
            self.postprocessed = True
            group.attrs['postprocessed'] = True
            self.file.flush()
        else:
            self.t = group['t']
            self.x_current = group['x_current']

    def apply_bc(self, i):
        # noinspection PyCallingNonCallable
        bc_value = self.bc_function(i * self.dt)
        if bc_value is not None:
            self.laser_energy_history[i] = bc_value
            self.electric_field[0, 1] = bc_value
            self.magnetic_field[0, 2] = bc_value / self.c
            # TODO: add polarization
            # self.electric_field[0, 2] = bc_value
            # self.magnetic_field[0, 1] = bc_value / self.c

    def init_solver(self):
        return self.solver.init_solver(self)

    def solve(self):
        return self.solver.solve(self)

    def direct_energy_calculation(self):
        r"""
        Direct energy calculation as

        :math:`E = \frac{\epsilon_0}{2} \sum_{i=0}^{NG} E^2 \Delta x`

        :return float E: calculated energy
        """
        return self.epsilon_0 * (self.electric_field ** 2).sum() * 0.5

    def gather_charge(self, list_species):
        # REFACTOR: move to Species
        self.charge_density[...] = 0.0
        for species in list_species:
            self.charge_density += species.gather_density() * species.eff_q
        # REFACTOR: optionally self.charge_density -= self.charge_density.mean() for periodic simulations

    def gather_current(self, list_species):
        # REFACTOR: move to Species
        self.current_density_x[...] = 0.0
        self.current_density_yz[...] = 0.0
        for species in list_species:
            self.current_longitudinal_gather_function(self.current_density_x, species.v[:, 0], species.x, self.dx, self.dt,
                                                      species.eff_q)
            self.current_transversal_gather_function(self.current_density_yz, species.v, species.x, self.dx, self.dt,
                                                     species.eff_q)

    def field_function(self, xp):
        result = self.interpolator(xp, np.hstack((self.electric_field, self.magnetic_field)), self.dx)
        return result[:, :3], result[:, 3:]

    def save_field_values(self, i):
        """Update the i-th set of field values, without those gathered from interpolation (charge\current)"""
        self.charge_density_history[i, :] = self.charge_density[:-1]
        self.current_density_history[i, :, 0] = self.current_density_x[1:-2]
        self.current_density_history[i, :, 1:] = self.current_density_yz[2:-2]
        self.electric_field_history[i] = self.electric_field[1:-1]
        self.magnetic_field_history[i] = self.magnetic_field[1:-1]


    def __str__(self):
        return f"NG{self.NG} dx{self.dx} NT {self.NT} dt {self.dt} c{self.c}eps{self.epsilon_0}"

def load_grid(file):
    """
    Loads grid data and create a Grid object.

    Parameters
    ----------
    file : h5py.File
        h5py file with data
    Returns
    -------
    Grid
        the loaded grid.
    """
    grid_data = file['grid']
    NG = grid_data.attrs['NGrid']
    L = grid_data.attrs['L']
    epsilon_0 = grid_data.attrs['epsilon_0']
    NT = grid_data['rho'].shape[0]
    c = grid_data.attrs['c']
    dx = grid_data.attrs['dx']
    dt = grid_data.attrs['dt']
    T = grid_data.attrs['T']
    periodic = grid_data.attrs['periodic']
    postprocessed = grid_data.attrs['postprocessed']
    print("file is", postprocessed)

    x = grid_data['x']
    grid = Grid(T = T,
                L = L,
                NG = NG,
                c = c,
                epsilon_0 = epsilon_0,
                periodic = periodic
                )
    grid.postprocessed = postprocessed
    grid.file = file
    assert grid.dx == dx
    assert grid.dt == dt
    assert grid.NT == NT
    assert np.allclose(x, grid.x)
    grid.charge_density_history = grid_data['rho']
    grid.current_density_history = grid_data['current']
    grid.electric_field_history = grid_data['Efield']
    grid.magnetic_field_history = grid_data['Bfield']
    grid.laser_energy_history = grid_data['laser']

    if not postprocessed:
        grid.postprocess()
    return grid

class TestGrid(Grid):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.charge_density_history = np.zeros((self.NT, self.NG))
        self.current_density_history = np.zeros((self.NT, self.NG, 3))
        self.electric_field_history = np.zeros((self.NT, self.NG, 3))
        self.magnetic_field_history = np.zeros((self.NT, self.NG, 3))
        self.laser_energy_history = np.zeros(self.NT, dtype=float)
    def postprocess(self, fourier=True):
        if not self.postprocessed:
            print("Postprocessing grid.")
            self.t = np.arange(self.NT) * self.dt
            if fourier:
                self.longitudinal_energy_history  = 0.5 * self.epsilon_0 * (self.electric_field_history[:,:,0] ** 2)
                perpendicular_electric_energy = 0.5 * self.epsilon_0 * (self.electric_field_history[:,:,1:] ** 2).sum(2) # over directions
                mu_zero_inv = 1/ (self.epsilon_0 * self.c**2)
                magnetic_energy = 0.5 * (self.magnetic_field_history **2).sum(2) * mu_zero_inv # over directions

                self.perpendicular_energy_history = perpendicular_electric_energy + magnetic_energy
                self.check_on_charge = np.gradient(self.electric_field_history[:, :, 0], self.dx, axis=1) * self.epsilon_0
                # fourier analysis
                from scipy import fftpack
                self.k_plot = fftpack.rfftfreq(int(self.NG), self.dx)[::2]
                self.longitudinal_energy_per_mode_history = np.abs(fftpack.rfft(self.longitudinal_energy_history))[:,::2]
                self.perpendicular_energy_per_mode_history = np.abs(fftpack.rfft(self.perpendicular_energy_history))[:,::2]

                self.longitudinal_energy_history  = self.longitudinal_energy_history.sum(1)
                self.perpendicular_energy_history = self.perpendicular_energy_history.sum(1)
                self.grid_energy_history = self.perpendicular_energy_history + self.longitudinal_energy_history # over positions
            vacuum_wave_impedance= 1/ (self.epsilon_0 * self.c)
            np.cumsum(self.laser_energy_history**2/ vacuum_wave_impedance * self.dt)
            self.x_current = self.x + self.dx / 2
            self.postprocessed = True
