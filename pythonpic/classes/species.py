"""Class representing a group of particles"""
# coding=utf-8
import numpy as np

from ..helper_functions.helpers import calculate_particle_snapshots, calculate_particle_iter_step, \
    is_this_saved_iteration, convert_global_to_particle_iter
from ..helper_functions.physics import gamma_from_v
from ..algorithms import density_profiles
from ..algorithms.particle_push import rela_boris_push
from scipy.stats import maxwell

MAX_SAVED_PARTICLES = int(1e4)

def n_saved_particles(n_p_available, n_upper_limit = MAX_SAVED_PARTICLES):
    """
    Calculates the number of saved particles from a dataset preventing it from
    hitting a predefined upper limit.

    Parameters
    ----------
    n_p_available : int
        number of particles in dataset
    n_upper_limit : int
        upper limit of particles that can be saved

    Returns
    -------
    save_every_n : int
        'step' between particles
    n_saved : int
        number of saved particles
    """

    if n_p_available <= n_upper_limit:
        return 1, n_p_available
    else:
        save_every_n = n_p_available // n_upper_limit + 1
        n_saved = np.ceil(n_p_available/save_every_n).astype(int)
        return save_every_n, n_saved

class Species:
    """
    Object representing a species of particles: ions, electrons, or simply
    a group of particles with a particular initial velocity distribution.

    Parameters
    ----------
    q : float
        particle charge
    m : float
        particle mass
    N : int
        number of macroparticles
    grid : Grid
        parent grid
    name : str
        name of group
    scaling : float
        number of particles represented by each macroparticle
    pusher : function
    individual_diagnostics : bool
        Set to `True` to save particle position and velocity
    """
    def __init__(self, q, m, N, grid, name="particles", scaling=1,
                 individual_diagnostics=False):
        self.q = q
        self.m = m
        self.N = int(N)
        self.N_alive = N
        self.scaling = scaling
        self.eff_q = q * scaling
        self.eff_m = m * scaling

        self.grid = grid
        self.grid.list_species.append(self)
        self.dt = grid.dt
        self.NT = grid.NT
        self.c = grid.c

        self.save_every_n_iterations = calculate_particle_iter_step(grid.NT)
        self.saved_iterations = calculate_particle_snapshots(grid.NT)
        self.x = np.zeros(N, dtype=np.float64)
        self.v = np.zeros((N, 3), dtype=np.float64)
        self.gathered_density = np.zeros(self.grid.NG+1, dtype=np.float64)
        self.energy = self.kinetic_energy
        self.alive = np.ones(N, dtype=bool)
        self.name = name
        self.save_every_n_particle, self.saved_particles = n_saved_particles(self.N, MAX_SAVED_PARTICLES)

        self.individual_diagnostics = individual_diagnostics
        if individual_diagnostics:
            self.position_history = np.zeros((self.saved_iterations, self.saved_particles), dtype=float)
            self.velocity_history = np.zeros((self.saved_iterations, self.saved_particles, 3), dtype=float)

        self.density_history = np.zeros((self.NT, self.grid.NG), dtype=float)
        self.velocity_mean_history = np.zeros((self.NT, 3), dtype=float)
        self.velocity_squared_mean_history = np.zeros((self.NT, 3), dtype=float)
        self.velocity_std_history = np.zeros((self.NT, 3), dtype=float)
        self.N_alive_history = np.zeros(self.NT, dtype=int)
        self.kinetic_energy_history = np.zeros(self.NT+1)

        self.postprocessed = False

    def prepare_history_arrays_h5py(self, f):
        """
        Prepares hdf5 history datasets in a given file.

        Parameters
        ----------
        f : h5py.File
        """

        self.file = f
        if "species" not in self.file:
            self.file.create_group("species")
        self.group = group = self.file["species"].create_group(self.name)
        if self.individual_diagnostics:
            self.position_history  = group.create_dataset(name="x", dtype=float, shape=(self.saved_iterations, self.saved_particles))
            self.velocity_history = group.create_dataset(name="v", dtype=float, shape=(self.saved_iterations, self.saved_particles, 3))
        self.density_history = group.create_dataset(name="density_history", dtype=float, shape=(self.NT, self.grid.NG))
        self.velocity_mean_history = group.create_dataset(name="v_mean", dtype=float, shape=(self.NT, 3))
        self.velocity_squared_mean_history = group.create_dataset(name="v2_mean", dtype=float, shape=(self.NT, 3))
        self.velocity_std_history = group.create_dataset(name="v_std", dtype=float, shape=(self.NT, 3))
        self.N_alive_history = group.create_dataset(name="N_alive_history", dtype=int, shape=(self.NT,))
        self.kinetic_energy_history = group.create_dataset(name="Kinetic energy", dtype=float, shape=(self.NT,))

        group.attrs['name'] = self.name
        group.attrs['N'] = self.N
        group.attrs['q'] = self.q
        group.attrs['m'] = self.m
        group.attrs['scaling'] = self.scaling
        group.attrs['postprocessed'] = self.postprocessed

    @property
    def gamma(self):
        """

        alculates the Lorentz factor from the current particle velocity.

        Returns
        -------
        gamma: numpy.ndarray
        """
        return gamma_from_v(self.v, self.c)

    @property
    def v_magnitude(self):
        """
        Calculates the magnitude of the velocity.

        Returns
        -------
        v: numpy.ndarray
        """
        return np.sqrt(np.sum(self.v**2, axis=1, keepdims=True))

    @property
    def momentum_history(self):
        return self.eff_m * np.array([gamma_from_v(v, self.c) * v for v in self.velocity_history])

    @property
    def kinetic_energy(self):
        return (self.gamma - 1).sum() * self.eff_m * self.c**2

    def velocity_push(self, field_function, time_multiplier=1):
        E, B = field_function(self.x)
        self.energy = rela_boris_push(self, E, time_multiplier * self.dt, B)

    def position_push(self):
        self.x += self.v[:, 0] * self.dt

    def gather_density(self):
        """A wrapper function to facilitate gathering particle density onto the grid.
        """
        self.gathered_density = self.grid.gather_density(self)
    """POSITION INITIALIZATION"""

    def distribute_uniformly(self, Lx: float, shift: float = 0, start_moat=0, end_moat=0):
        """

        Distribute uniformly on grid.

        Parameters
        ----------
        Lx : float
            physical grid size
        shift : float
            a constant displacement for all particles
        start_moat : float
            left boundary size
        end_moat :
            right boundary size

        """
        self.x = (np.linspace(start_moat + Lx / self.N * 1e-10, Lx - end_moat, self.N,
                              endpoint=False) + shift * self.N / Lx / 10) % Lx  # Type:

    def distribute_nonuniformly(self, moat_length, ramp_length, plasma_length,
                                resolution_increase=1000, profile="linear"):
        dense_x = np.linspace(moat_length*0.95, (moat_length + plasma_length)*1.05, self.N * resolution_increase)
        self.x = density_profiles.generate(dense_x, density_profiles.FDENS, moat_length,
                                           ramp_length,
                                           plasma_length, self.N, profile)

    def sinusoidal_position_perturbation(self, amplitude: float, mode: int):
        """
        Displace positions by a sinusoidal perturbation calculated for each particle.

        ..math:
            dx = amplitude * cos(2 * mode * pi * x / L)

        Parameters
        ----------
        amplitude : float
        mode : int, float


        """
        self.x += amplitude * np.cos(2 * mode * np.pi * self.x / self.grid.L) # TODO: remove 2*

    def random_position_perturbation(self, std: float):
        """
        Displace positions by gaussian noise. May reduce number of particles afterwards due to applying BC.

        Parameters
        ----------
        std : float
            standard deviation of the noise, in units of grid cell size
        Returns
        -------

        """
        self.x += np.random.normal(scale=std*self.grid.dx, size=self.N)

    def random_velocity_init(self, amplitude: float):
        random_theta = np.random.random(size=self.N) * 2 * np.pi
        random_phi = np.random.random(size=self.N) * np. pi
        directions_x = np.cos(random_theta) * np.sin(random_phi)
        directions_y = np.sin(random_theta) * np.sin(random_phi)
        directions_z = np.cos(random_phi)
        amplitudes = maxwell.rvs(size=self.N, loc=amplitude)
        self.v[:,0] += amplitudes * directions_x
        self.v[:,1] += amplitudes * directions_y
        self.v[:,2] += amplitudes * directions_z


    """VELOCITY INITIALIZATION"""

    def sinusoidal_velocity_perturbation(self, axis: int, amplitude: float, mode: int):
        """
        Displace velocities by a sinusoidal perturbation calculated for each particle.

        Parameters
        ----------
        axis : int
            direction, for 3d velocities
        amplitude : float
        mode : int


        """
        self.v[:, axis] += amplitude * np.cos(2 * mode * np.pi * self.x / self.grid.L)

    def random_velocity_perturbation(self, axis: int, std: float):
        """
        Add Gausian noise to particle velocities on

        Parameters
        ----------
        axis :  int
            direction, for 3d velocities
        std : float
            standard deviation of perturbation


        """
        self.v[:, axis] += np.random.normal(scale=std, size=self.N)

    # def init_velocity_maxwellian(self, T, resolution_increase = 1000):
    #     thermal_velocity = 1
    #     dense_p = np.linspace(0, 4 * thermal_velocity, self.N/4 * 1000)
    #
    #     # TODO: WORK IN PROGRESS
    #     self.v = result

    """ DATA ACCESS """

    def save_particle_values(self, i: int):
        """
        Update the i-th set of saved particle values (positions, velocities)
        and densities on the grid.

        Parameters
        ----------
        i : int
        """
        N_alive = self.x.size
        self.density_history[i] = self.gathered_density[:-1]
        if self.individual_diagnostics and is_this_saved_iteration(i, self.save_every_n_iterations):
            save_every_n_particle, saved_particles = n_saved_particles(N_alive, self.saved_particles)

            # print(f"out of {N_alive} save every {save_every_n_particle} with mean x {self.x.mean()}")
            index = convert_global_to_particle_iter(i, self.save_every_n_iterations)
            try:
                self.position_history[index, :saved_particles] = self.x[::save_every_n_particle]
                self.velocity_history[index, :saved_particles] = self.v[::save_every_n_particle]
            except ValueError:
                data = N_alive, save_every_n_particle, saved_particles, self.N, self.x.size
                raise ValueError(data)
        self.N_alive_history[i] = N_alive
        if N_alive > 0:
            self.velocity_mean_history[i] = self.v.mean(axis=0)
            self.velocity_squared_mean_history[i] = (self.v**2).mean(axis=0)
            self.velocity_std_history[i] = self.v.std(axis=0)
        self.kinetic_energy_history[i] = self.energy



    def postprocess(self):
        """
        Perform postprocessing on the `Species`. At the moment, this simply
        scales the density of macroparticles to the density of real particles.
        """
        if not self.postprocessed:
            print(f"Postprocessing {self.name}.")
            self.density_history[...] *= self.scaling
            self.postprocessed = self.group.attrs['postprocessed'] = True
            self.file.flush()

    def __repr__(self, *args, **kwargs):
        return f"Species(q={self.q:.4f},m={self.m:.4f},N={self.N},name=\"{self.name}\",NT={self.NT})"

    def __str__(self):
        return f"{self.N} {self.scaling:.2e}-{self.name} with q = {self.q:.2e} C, m = {self.m:.2e} kg," \
               f" {self.saved_iterations} saved history " \
               f"steps over {self.NT} iterations"

def load_species(f, grid):
    """
    Loads species data from h5py file.

    Parameters
    ----------
    f : `h5py.File`
        Data file
    grid : `Grid`
        grid to load particles onto

    Returns
    -------
    list_species : list
    """
    # TODO: could do a for loop here to load multiple species
    list_species = []
    for name in f['species']:
        species_data = f['species'][name]
        N = species_data.attrs['N']
        q = species_data.attrs['q']
        m = species_data.attrs['m']
        scaling = species_data.attrs['scaling']
        postprocessed = species_data.attrs['postprocessed']


        species = Species(q, m, N, grid, name, scaling, individual_diagnostics=False)
        species.velocity_mean_history = species_data["v_mean"]
        species.velocity_squared_mean_history = species_data["v2_mean"]
        species.velocity_std_history = species_data["v_std"]
        species.density_history = species_data["density_history"]
        species.file = f
        species.group = species_data
        species.postprocessed = postprocessed


        if "x" in species_data and "v" in species_data:
            species.individual_diagnostics = True
            species.position_history = species_data["x"]
            species.velocity_history = species_data["v"]
        species.N_alive_history = species_data["N_alive_history"]
        species.kinetic_energy_history = species_data["Kinetic energy"]
        if not postprocessed:
            species.postprocess()
        list_species.append(species)
    return list_species



class TestSpecies(Species):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.individual_diagnostics:
            self.position_history = np.zeros((self.saved_iterations, self.saved_particles), dtype=float)
            self.velocity_history = np.zeros((self.saved_iterations, self.saved_particles, 3), dtype=float)

        self.density_history = np.zeros((self.NT, self.grid.NG), dtype=float)
        self.velocity_mean_history = np.zeros((self.NT, 3), dtype=float)
        self.velocity_squared_mean_history = np.zeros((self.NT, 3), dtype=float)
        self.velocity_std_history = np.zeros((self.NT, 3), dtype=float)
        self.N_alive_history = np.zeros(self.NT, dtype=int)
        self.kinetic_energy_history = np.zeros(self.NT+1)

class Particle(TestSpecies):
    """
    A helper class for quick creation of a single particle for test purposes.
    Parameters
    ----------
    grid : Grid
        parent grid
    x : float
        position
    vx : float
        x velocity
    vy : float
        y velocity
    vz : float
        z velocity
    q : float
        particle charge
    m : float
        particle mass
    name : str
        name of group
    scaling : float
        number of particles per macroparticle
    pusher : function
        particle push algorithm
    """
    def __init__(self, grid, x, vx, vy=0, vz=0, q=1, m=1, name="Test particle", scaling=1):
        # noinspection PyArgumentEqualDefault
        super().__init__(q, m, 1, grid, name, scaling = scaling,
                         individual_diagnostics=True)
        self.x[:] = x
        self.v[:, 0] = vx
        self.v[:, 1] = vy
        self.v[:, 2] = vz


