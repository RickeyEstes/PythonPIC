"""Data interface class"""
# coding=utf-8
import os
import time

import h5py
import numpy as np
import matplotlib.pyplot as plt

from .grid import Grid, load_grid
from .species import load_species
from ..helper_functions.helpers import report_progress, git_version, config_filename
from ..visualization import animation, static_plots


current_time = time.strftime("%Y-%m-%d %H:%M")
current_time_filename = time.strftime("%Y-%m-%d_%H-%M-%S")
class Simulation:
    """

    Contains data from one run of the simulation.

    Parameters
    ----------
    grid : Grid
    list_species : list
    run_date : str
    git_ver : str
    filename : str
    title : str
    """
    def __init__(self, grid: Grid, list_species=None, run_date=current_time, git_version=git_version(),
                 filename=current_time_filename, category_type=None, config_version=None, title="",
                 considered_large=False):
        self.NT = grid.NT
        self.dt = grid.dt
        self.t = np.arange(self.NT) * self.dt

        self.grid = grid
        if list_species is None:
            list_species = []
        self.list_species = list_species

        self.filename = config_filename(filename, category_type, config_version)
        self.title = title
        self.git_version = git_version
        self.run_date = run_date

        self.postprocessed=False
        self.runtime = None
        self.considered_large = considered_large

    def postprocess(self):
        if not self.postprocessed:
            self.grid.postprocess()
            self.total_kinetic_energy = np.zeros(self.NT)
            for species in self.list_species:
                species.postprocess()
                self.total_kinetic_energy += species.kinetic_energy_history
            print("Postprocessing simulation.")
            self.total_energy_history =  self.total_kinetic_energy + self.grid.grid_energy_history[...]
            self.postprocessed = True
        # TODO: this doesn't save
        return self

    def grid_species_initialization(self):
        """
        Initializes grid and particle relations:
        1. gathers charge from particles to grid
        2. solves Poisson equation to get initial field
        3. initializes pusher via a step back
        """
        self.grid.apply_bc(0)
        for species in self.list_species:
            species.velocity_push(self.grid.field_function, -0.5)
        self.grid.gather_charge(self.list_species)
        self.grid.gather_current(self.list_species)
        self.grid.init_solve()
        for species in self.list_species:
            species.position_push()
            self.grid.apply_particle_bc(species)
        return self

    def iteration(self, i: int):
        """

        :param int i: iteration number
        Runs an iteration step
        1. saves field values
        2. for all particles:
            2. 1. saves particle values
            2. 2. pushes particles forward

        """
        self.grid.save_field_values(i)  # CHECK: is this the right place, or after loop?
        self.grid.apply_bc(i)
        for species in self.list_species:
            species.velocity_push(self.grid.field_function) # TODO should be inplace?
        self.grid.gather_charge(self.list_species)
        self.grid.gather_current(self.list_species)
        self.grid.solve()
        for species in self.list_species:
            species.position_push()
            species.save_particle_values(i)
            self.grid.apply_particle_bc(species)

    def iteration_lite(self, i: int):
        """

        :param int i: iteration number
        Runs an iteration step
        1. saves field values
        2. for all particles:
            2. 1. saves particle values
            2. 2. pushes particles forward

        """
        self.grid.apply_bc(i)
        for species in self.list_species:
            species.velocity_push(self.grid.field_function) # TODO should be inplace?
        self.grid.gather_charge(self.list_species)
        self.grid.gather_current(self.list_species)
        self.grid.solve()
        for species in self.list_species:
            species.position_push()
            self.grid.apply_particle_bc(species)

    def run(self, init=True):
        """
        Run n iterations of the simulation, saving data as it goes.

        Also measures runtime, saving it in self.runtime as a float with units of seconds.

        Parameters
        ----------
        init : bool
            Whether or not to initialize the simulation (particle placement, grid interaction).
            Not necessary, for example, in some tests.

        Returns
        -------
        self: Simulation
            The simulation, for chaining purposes.
        """
        try:
            if not os.path.exists(os.path.dirname(self.filename)):
                os.makedirs(os.path.dirname(self.filename))
            self.grid_file = h5py.File(self.filename, "w")
            for species in self.list_species:
                species.prepare_history_arrays_h5py(self.grid_file)
            self.grid.prepare_history_arrays_h5py(self.grid_file)
            if init:
                self.grid_species_initialization()
            start_time = time.time()
            for i in range(self.NT):
                if self.considered_large and i % (self.NT // 100) == 0:
                    report_progress(i, self.NT, start_time)
                self.iteration(i)
            self.runtime = time.time() - start_time
            return self
        except KeyboardInterrupt:
            self.grid_file.close()
            print("Simulation interrupted. Removing data.")
            os.remove(self.filename)
            exit()

    def run_lite(self, init=True):
        """
        Run n iterations of the simulation, saving data as it goes.

        Also measures runtime, saving it in self.runtime as a float with units of seconds.

        Parameters
        ----------
        init : bool
            Whether or not to initialize the simulation (particle placement, grid interaction).
            Not necessary, for example, in some tests.

        Returns
        -------
        self: Simulation
            The simulation, for chaining purposes.
        """
        if init:
            self.grid_species_initialization()
        start_time = time.time()
        for i in range(self.NT):
            self.iteration_lite(i)
        self.runtime = time.time() - start_time
        return self.runtime

    def lazy_run(self):
        """Does a simulation run() unless there's already a saved data with that file.

        If that file contains the same initial conditions and config version, the simulation's results are
        loaded instead.

        If the simulation errors during loading, it is ran anew."""
        print(f"Path is {self.filename}")
        file_exists = os.path.isfile(self.filename)
        if file_exists:
            print("Found file. Attempting to load...")
            try:
                loaded = load_simulation(self.filename)
                print("Managed to load file.")
                if loaded == self:
                    return loaded.postprocess()
                else:
                    print("Simulation files differ.")
            except KeyError as err:
                print(err)
        print("Running simulation")
        return self.run().save_data().postprocess()

    def test_run(self):
        """Does a blind run without saving data, for test purposes."""
        try:
            os.remove(self.filename)
        except FileNotFoundError:
            pass
        result = self.run().postprocess()
        return result

    # noinspection PyUnusedLocal
    def plots(self,
              show_static: bool = False,
              save_static: bool = False,
              show_animation: bool = False,
              save_animation: bool = False,
              snapshot_animation: bool = False,
              alpha: float = 0.7,
              animation_type=animation.FullAnimation,
              static_type = static_plots.static_plots_large,
              frames="few"
              ):
        """
        Wrapper to run visual analysis on saved hdf5 file. Displays static plots
        and animations.

        Parameters
        ----------
        show_static : bool
        save_static : bool
        show_animation : bool
        save_animation : bool
        snapshot_animation : bool
        alpha : float
            Used for opacity in plots
        animation_type : `pythonpic.visualization.Animation`
        frames : str
            see docs of `pythonpic.visualization.Animation`
        static_type

        Returns
        -------

        """
        if "DISPLAY" not in os.environ.keys():
            print("Can't plot, DISPLAY not defined!")
            return False
        if show_static or show_animation or save_animation or save_static or snapshot_animation:
            self.postprocess()
            if show_animation or save_animation or snapshot_animation:
                anim = animation_type(self, alpha, frames)
                if snapshot_animation:
                    anim.snapshot_animation()
                if save_animation or show_animation:
                    anim_object = anim.full_animation(save_animation)
            if save_static or show_static:
                filename = self.filename.replace(".hdf5", ".png") if save_static else None
                static = static_type(self, filename)
                if not show_static:
                    plt.close(static)
            if show_animation or show_static:
                try:
                    plt.show()
                except KeyboardInterrupt:
                    print("Quitting.")
                    plt.close("all")
            else:
                plt.close("all")


    def plots_1d(self, *args, **kwargs):
        self.plots(*args, **kwargs, animation_type =
            animation.OneDimAnimation, static_type=static_plots.electrostatic_static_plots)

    def plots_3d(self, *args, **kwargs):
        self.plots(*args, **kwargs, animation_type =
            animation.FullAnimation, static_type=static_plots.static_plots)

    def phase_1d(self, *args, **kwargs):
        self.plots(*args, **kwargs, animation_type =
            animation.OneDimPhaseAnim, static_type=static_plots.static_plots)

    def grid_1d(self, *args, **kwargs):
        self.plots(*args, **kwargs, animation_type =
            animation.OneDimGridAnim, static_type=static_plots.static_plots)

    def wave_1d(self, *args, **kwargs):
        self.plots(*args, **kwargs, animation_type =
            animation.OneDimFieldAnim, static_type=static_plots.static_plots)

    def save_data(self):
        """Save simulation data to hdf5.
        filename by default is the timestamp for the simulation."""
        if not os.path.exists(os.path.dirname(self.filename)):
            os.makedirs(os.path.dirname(self.filename))
        f = self.grid_file


        f.attrs['dt'] = self.dt
        f.attrs['NT'] = self.NT
        f.attrs['run_date'] = self.run_date
        f.attrs['git_version'] = self.git_version
        f.attrs['title'] = self.title
        f.attrs['runtime'] = self.runtime
        f.attrs['considered_large'] = self.considered_large
        self.grid_file.flush()
        print(f"Saved file to {self.filename}")
        return self

    def __str__(self, *args, **kwargs):
        filename = os.path.basename(self.filename)
        result_string = f"""
        {self.title} simulation ({filename}) lasting {self.NT} iterations, dt = {self.dt:.3e} s
        Done on {self.run_date} from git version {self.git_version}
        {self.grid}
        """.strip()
        for i, species in enumerate(self.list_species):
            result_string += f"\nSpecies {i+1}: {species}"
        return result_string  # REFACTOR: add information from config file (run_coldplasma...)

    def __eq__(self, other):
        return True # TODO: compare

def load_simulation(filename: str):
    """
    Create a Simulation object from a hdf5 file.

    Parameters
    ----------
    filename : str
        Path to a hdf5 file.

    Returns
    -------
    Simulation
    """
    f = h5py.File(filename, "r+")
    title = f.attrs['title']
    grid = load_grid(f)

    all_species = load_species(f, grid)
    run_date = f.attrs['run_date']
    git_version = f.attrs['git_version']
    considered_large = f.attrs['considered_large']
    S = Simulation(grid, all_species, run_date=run_date, git_version=git_version, filename=filename, title=title, considered_large=considered_large)
    S.filename = filename

    S.postprocess()
    return S
