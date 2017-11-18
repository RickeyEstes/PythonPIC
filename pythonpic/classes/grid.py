"""The spatial grid"""
# coding=utf-8
import numpy as np
import h5py
import scipy.fftpack as fft
from scipy.integrate import cumtrapz, trapz

from ..helper_functions import physics
from ..algorithms import charge_deposition, FieldSolver, BoundaryCondition, \
    current_deposition, field_interpolation
from ..algorithms.FieldSolver import (BunemanLongitudinalSolver,
                                    BunemanTransversalSolver,
                                    FourierLongitudinalSolver)


class Grid:
    """
    Abstract class for an object representing the Eulerian grid on which
    charges, currents and fields are computed and stored.

    Actual simulations should be ran on `PeriodicGrid` or `NonperiodicGrid`.

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
    bc : `BoundaryCondition`
    """

    def __init__(self, T: float, L: float, NG: int, c: float = 1,
                 epsilon_0: float = 1, bc=BoundaryCondition.BC()):

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

        self.bc = bc
        self.k = 2 * np.pi * fft.fftfreq(self.NG, self.dx)
        self.k[0] = 0.0001

        self.list_species = []
        self.postprocessed = False
        self.postprocessed_fourier = False
        self.periodic = None

    def prepare_history_arrays_h5py(self, f):
        """
        Prepares hdf5 history datasets in a given file.

        Parameters
        ----------
        f : h5py.File
        """
        self.file = f
        group = self.file.create_group("grid")
        self.charge_density_history = group.create_dataset(name="rho", dtype=float, shape=(self.NT, self.NG))
        self.current_density_history = group.create_dataset(name="current", dtype=float, shape=(self.NT, self.NG, 3))
        self.electric_field_history = group.create_dataset(name="Efield", dtype=float, shape=(self.NT, self.NG, 3))
        self.magnetic_field_history = group.create_dataset(name="Bfield", dtype=float, shape=(self.NT, self.NG, 3))
        self.laser_energy_history = group.create_dataset(name="laser", dtype=float, shape=(self.NT,))
        group.create_dataset(name="x", dtype=float, data=self.x)

        h5py_dictionary = {'NGrid':                 self.NG,
                           'L':                     self.L,
                           'epsilon_0':             self.epsilon_0,
                           'c':                     self.c,
                           'dt':                    self.dt,
                           'dx':                    self.dx,
                           'NT':                    self.NT,
                           'T':                     self.T,
                           'periodic':              self.periodic,
                           'postprocessed':         self.postprocessed,
                           'postprocessed_fourier': self.postprocessed_fourier
                           }
        for key, value in h5py_dictionary.items():
            group.attrs[key] = value

    def postprocess(self):
        """
        Runs postprocessing on the grid if it has not been postprocessed yet.

        Postprocessing includes calculating field energies from saved fields,
        et cetera.

        This saves analysis results to hdf5, so the dataset's size will grow.
        """
        group = self.file['grid']
        if not self.postprocessed:
            print("Postprocessing grid.")
            self.t = group.create_dataset(name="t", data=np.arange(self.NT) * self.dt)
            vacuum_wave_impedance= 1/ (self.epsilon_0 * self.c)
            self.laser_energy_history[...] = np.cumsum(self.laser_energy_history[...]**2/ vacuum_wave_impedance * self.dt)


            mu_zero_inv = 1/ (self.epsilon_0 * self.c**2)
            poynting = (self.electric_field_history[:, :, 1] * self.magnetic_field_history[:, :, 2] +
                        self.electric_field_history[:, :, 2] * self.magnetic_field_history[:, :, 1]) * mu_zero_inv / self.dx

            self.poynting_history = group.create_dataset(name="poynting", data=poynting)
            integrand = self.poynting_history[:, 0] - self.poynting_history[:, -1]
            self.energy_via_bc_history = group.create_dataset(name="energy_poynting_bc", data=cumtrapz(integrand, self.t[...], initial=0))
            self.entering_energy_via_bc_history = group.create_dataset(name="entering_energy_poynting_bc", data=cumtrapz(self.poynting_history[:,0], self.t[...], initial=0))
            self.x_current = group.create_dataset(name="x_current", data=self.x + self.dx / 2)
            self.postprocessed = True
            group.attrs['postprocessed'] = True

            self.longitudinal_energy_history  = group.create_dataset("longitudinal_energy", data=0.5 * self.epsilon_0 * (self.electric_field_history[:,:,0] ** 2))
            perpendicular_electric_energy = 0.5 * self.epsilon_0 * (self.electric_field_history[:,:,1:] ** 2).sum(2) # over directions
            magnetic_energy = 0.5 * (self.magnetic_field_history[...] **2).sum(2) * mu_zero_inv # over directions

            self.perpendicular_energy_history = group.create_dataset("perpendicular_energy", data=perpendicular_electric_energy + magnetic_energy)
            self.check_on_charge = group.create_dataset("charge_test", data=np.gradient(self.electric_field_history[:, :, 0], self.dx, axis=1) * self.epsilon_0)
            # fourier analysis
            from scipy import fftpack
            self.k_plot = group.create_dataset("k_plot", data=fftpack.rfftfreq(int(self.NG), self.dx)[::2])
            self.longitudinal_energy_per_mode_history = group.create_dataset("longitudinal_fourier", data=np.abs(fftpack.rfft(self.longitudinal_energy_history))[:,::2])
            self.perpendicular_energy_per_mode_history = group.create_dataset("perpendicular_fourier", data=np.abs(fftpack.rfft(self.perpendicular_energy_history))[:,::2])

            self.longitudinal_energy_history  = group.create_dataset("total_longitudinal", data=trapz(self.electric_field_history[:,:,0]**2 * self.epsilon_0 / 2 / self.dx, self.x, axis=1))
            self.perpendicular_energy_history = group.create_dataset("total_perpendicular", data=self.perpendicular_energy_history[...].sum(1))
            self.grid_energy_history = group.create_dataset("total_grid", data=self.perpendicular_energy_history[...] + self.longitudinal_energy_history[...])
            group.attrs["postprocessed_fourier"] = True
            self.postprocessed_fourier = True
            self.file.flush()
        else:
            self.t = group['t']
            self.x_current = group['x_current']
            self.poynting_history = group['poynting']
            self.energy_via_bc_history = group["energy_poynting_bc"]
            self.entering_energy_via_bc_history = group["entering_energy_poynting_bc"]
            self.perpendicular_energy_history = group["perpendicular_energy"]
            self.check_on_charge = group["charge_test"]
            self.k_plot = group["k_plot"]
            self.longitudinal_energy_per_mode_history = group["longitudinal_fourier"]
            self.perpendicular_energy_history = group["perpendicular_fourier"]
            self.longitudinal_energy_history = group["total_longitudinal"]
            self.perpendicular_energy_history = group["total_perpendicular"]
            self.grid_energy_history = group["total_grid"]

    def apply_bc(self, i):
        """
        Applies boundary conditions at a given iteration, modifying fields in
        place.

        Parameters
        ----------
        i : int
            Iteration numbed
        """
        self.bc.apply(self.electric_field, self.magnetic_field, i * self.dt)
        self.laser_energy_history[i] = np.sqrt(np.sum(self.electric_field[self.bc.index, 1:]**2))

    def init_solve(self, neutralize = False):
        """
        Performs the initial, spectral iteration of the field solver.
        See `FieldSolver` for details.
        """
        self.electric_field[1:-1, 0] = FourierLongitudinalSolver(
            self.charge_density[:-1], self.k, epsilon_0=self.epsilon_0, neutralize=neutralize
            )

        BunemanTransversalSolver(self.electric_field,
                                        self.magnetic_field,
                                        self.current_density_yz, self.dt,
                                        self.c, self.epsilon_0)


    def solve(self):
        """
        Performs an iteration of the field solver. This is based on the
        rotation equations. See `FieldSolver` for details.
        Returns
        -------

        """
        BunemanLongitudinalSolver(self.electric_field, self.current_density_x,
                                                              self.dt,
                                                              self.epsilon_0,
                                                              )
        BunemanTransversalSolver(self.electric_field,
                                       self.magnetic_field,
                                       self.current_density_yz, self.dt,
                                       self.c, self.epsilon_0)

    def direct_energy_calculation(self):
        r"""
        Direct energy calculation as

        :math:`E = \frac{\epsilon_0}{2} \sum_{i=0}^{NG} E^2 \Delta x`

        :return float E: calculated energy
        """
        return self.epsilon_0 * (self.electric_field ** 2).sum() * 0.5

    def gather_charge(self, list_species, force_periodic=False):
        """
        Gathers charge onto the Eulerian grid.

        Parameters
        ----------
        list_species : list
            A list of species to gather charge from.
        force_periodic : bool

        """
        self.charge_density[...] = 0.0
        for species in list_species:
            self.charge_density += species.gather_density() * species.eff_q
        if self.periodic and force_periodic:
            self.charge_density -= self.charge_density.mean()

    def gather_current(self, list_species):
        """
        Gathers transversal and longitudinal current onto the Eulerian grid.

        Parameters
        ----------
        list_species : list
            A list of species to gather charge from.
        """
        self.current_density_x[...] = 0.0
        self.current_density_yz[...] = 0.0
        for species in list_species:
            self.current_gather_function(self.current_density_x,
                                         self.current_density_yz,
                                         species.v, species.x,
                                         self.dx, self.dt, species.eff_q)

    def field_function(self, xp):
        """
        Interpolates fields to particle locations.

        Parameters
        ----------
        xp : `numpy.ndarray`
            Particle positions.

        Returns
        -------
        E, B : `numpy.ndarray`


        """
        result = self.interpolator(xp, np.hstack((self.electric_field,
                                                  self.magnetic_field)), self.dx)
        return result[:, :3], result[:, 3:]

    def save_field_values(self, i):
        """
        Update the i-th set of historical grid quantity values - charge,
        currents and fields.

        This doesn't save quantities at the grid boundaries, also known
        as guard cells!

        Parameters
        ----------
        i : int
            iteration number

        """
        self.charge_density_history[i, :] = self.charge_density[:-1]
        self.current_density_history[i, :, 0] = self.current_density_x[1:-2]
        self.current_density_history[i, :, 1:] = self.current_density_yz[2:-2]
        self.electric_field_history[i] = self.electric_field[1:-1]
        self.magnetic_field_history[i] = self.magnetic_field[1:-1]

    def __repr__(self):
        return f"Grid(T={self.T}, L={self.L}, NG={self.NG}, c={self.c}," \
               f" epsilon_0={self.epsilon_0}, periodic={self.periodic}," \
               f" dt={self.dt}, dx={self.dx}"

    def __str__(self):
        return f"{self.NG}-cell grid of length {self.L:.2f} m." \
               f" $\\varepsilon_0 = {self.epsilon_0}$ F/m," \
               f" $c = {self.c:.2e}$ m/s"


class PeriodicGrid(Grid):
    """
    Object representing a periodic Eulerian grid on which charges, currents and
    fields are computed and stored.

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
    bc : `BoundaryCondition`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.charge_gather_function = charge_deposition.periodic_density_deposition
        self.current_gather_function = current_deposition \
            .periodic_current_deposition
        self.current_gather_function = current_deposition.periodic_current_deposition
        self.particle_bc = BoundaryCondition.return_particles_to_bounds
        self.interpolator = field_interpolation.PeriodicInterpolateField
        self.periodic = True


class NonperiodicGrid(Grid):
    """
    Object representing a non-periodic Eulerian grid on which charges, currents
    and fields are computed and stored.

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
    bc : `BoundaryCondition`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.charge_gather_function = charge_deposition.density_deposition
        self.current_gather_function = current_deposition.aperiodic_current_deposition
        self.current_gather_function = current_deposition.aperiodic_current_deposition
        self.particle_bc = BoundaryCondition.kill_particles_outside_bounds
        self.interpolator = field_interpolation.AperiodicInterpolateField
        self.periodic = False


class PeriodicTestGrid(PeriodicGrid):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs),
        self.charge_density_history = np.zeros((self.NT, self.NG))
        self.current_density_history = np.zeros((self.NT, self.NG, 3))
        self.electric_field_history = np.zeros((self.NT, self.NG, 3))
        self.magnetic_field_history = np.zeros((self.NT, self.NG, 3))
        self.laser_energy_history = np.zeros(self.NT, dtype=float)

    def postprocess(self):
        if not self.postprocessed:
            print("Postprocessing grid.")
            self.t = np.arange(self.NT) * self.dt
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


class NonperiodicTestGrid(NonperiodicGrid):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs),
        self.charge_density_history = np.zeros((self.NT, self.NG))
        self.current_density_history = np.zeros((self.NT, self.NG, 3))
        self.electric_field_history = np.zeros((self.NT, self.NG, 3))
        self.magnetic_field_history = np.zeros((self.NT, self.NG, 3))
        self.laser_energy_history = np.zeros(self.NT, dtype=float)
    def postprocess(self):
        if not self.postprocessed:
            print("Postprocessing grid.")
            self.t = np.arange(self.NT) * self.dt
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


def load_grid(file):
    """
    Loads grid data from a h5py file and creates a Grid object to allow easy
    recreation of saved simulations.

    Parameters
    ----------
    file : h5py.File

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

    x = grid_data['x']
    if periodic:
        grid_type = PeriodicGrid
    else:
        grid_type = NonperiodicGrid
    grid = grid_type(T=T, L=L, NG=NG, c=c, epsilon_0=epsilon_0)
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
