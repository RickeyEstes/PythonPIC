# coding=utf-8
import argparse
import errno
import os
import time
# from ..classes import load_simulation
import subprocess

import numpy as np


def config_filename(run_name, category_name=None, version_number=None):
    """
    Prepares config filename for saving.

    Parameters
    ----------
    run_name : str

    category_name : str

    version_number : int


    Examples
    ----------
    >>> config_filename("run")
    'data_analysis/run/run.hdf5'
    >>> config_filename("run", "simulation_type")
    'data_analysis/simulation_type/run/run.hdf5'
    >>> config_filename("run", "simulation_type", 1)
    'data_analysis/simulation_type/v1/run/run.hdf5'

    Returns
    -------

    """
    return "data_analysis/" + \
           f"{category_name+'/' if category_name else ''}" + \
           f"{'v' + str(version_number) + '/' if version_number else ''}" + \
           f"{run_name}/{run_name}.hdf5"


#
def report_progress(i: int, NT: int, beginning_time = None):
    """
    Prints out a message on how many iterations out of how many total have been completed.

    Parameters
    ----------
    i : int
        Current iteration number
    NT : int
        Total iteration number
    beginning_time : `float` or `bool`
        `time.time()`-generated runtime at beginning of simulation instance.
        Used for estimating runtime. Default is `None`.
    Examples
    ----------
    >>> report_progress(0, 128)
    0/128 iterations (0%) done!
    >>> report_progress(33, 200)
    33/200 iterations (16%) done!
    >>> report_progress(200, 200)
    200/200 iterations (100%) done!

    """
    start_string = f"{i}/{NT} iterations ({i/NT*100:.0f}%) done!"
    if beginning_time and i > 0:
        iterations_left = NT - i
        time_delta = time.time() - beginning_time
        time_per_iteration = time_delta / i
        estimated_remaining_time = iterations_left * time_per_iteration
        start_string += f" Current runtime: {time_delta:.0f}s; {estimated_remaining_time:.0f}s estimated left."
    print(start_string)


def git_version() -> str:
    """
    Returns the current git version identifier.
    -------
    str
        The current seven first characters of the current git version hash.
    """
    return subprocess.check_output(['git', 'describe', '--always']).decode()[:-1]


def calculate_particle_iter_step(NT, f=np.log2):
    """
    Calculate number of iterations between saving particle data.

    The function is meant to be easy to change.
    It should, however, rise slower than :math:`f(x) = x`.
    Good candidates are logarithms and roots.

    If the result is lower than 1, it returns 1.

    Parameters
    ----------
    NT : int
        total number of iterations
    f : function
        A function of a single variable returning a single variable

    Examples
    ----------
    >>> calculate_particle_iter_step(128, np.log2)
    7
    >>> calculate_particle_iter_step(128, np.sqrt)
    11
    >>> calculate_particle_iter_step(128, np.log10)
    2
    >>> calculate_particle_iter_step(3, np.log10)
    1

    Returns
    -------
    int
        iteration step capped
    """
    result = int(f(NT))
    return result if result > 1 else 1


def calculate_particle_snapshots(NT, f = np.log2):
    """
    Calculates number of particle snapshots via `calculate_particle_iter_step`. See docs of that.

    Parameters
    ----------
    NT : int
        total number of iterations
    f : function
        A slowly rising function of a single variable returning a single variable

    Examples
    ----------
    >>> calculate_particle_snapshots(128, np.log2)
    19
    >>> calculate_particle_snapshots(128, np.sqrt)
    12
    >>> calculate_particle_snapshots(3, np.log10)
    4

    Returns
    -------
    int
        number of iteration steps to be saved.

    """
    return int(NT / calculate_particle_iter_step(NT, f)) + 1 # CHECK if result shouldn't be as NT, so remove + 1 here


def is_this_saved_iteration(i, save_every_n_iterations):
    return i % save_every_n_iterations == 0


def convert_global_to_particle_iter(i, save_every_n_iterations):
    return i // save_every_n_iterations


def plotting_parser(description):
    """
    Parses flags for showing or animating plots

    :param str description: Short program description
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--show-static", help="Show plots once the run finishes", action="store_true")
    parser.add_argument("--save-static", help="Save plots once the run finishes", action="store_true")
    parser.add_argument("--show-animation", help="Show the animation", action="store_true")
    parser.add_argument("--save-animation", help="Save the animation", action="store_true")
    parser.add_argument("--snapshot-animation", help="Save the animation as snapshots", action="store_true")
    args = parser.parse_args()
    return args.show_static, args.save_static, args.show_animation, args.save_animation, args.snapshot_animation


def make_sure_path_exists(path):
    try:
        os.makedirs(os.path.dirname(path))
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


colors = "brgyc"
directions = "xyz"