"""plotting suite for simulation data analysis, can be called from command line"""
# coding=utf-8
import argparse
import os

import matplotlib.pyplot as plt

from ..classes import simulation
from ..visualization import animation
from ..visualization import static_plots

directory = "data_analysis/"


def plots(file,
          show_static: bool = False,
          save_static: bool = False,
          show_animation: bool = False,
          save_animation: bool = False,
          snapshot_animation: bool = False,
          alpha: float = 0.7,
          animation_type=animation.FullAnimation,
          frames="few"
          ):
    """
    Wrapper to run visual analysis on saved hdf5 file. Displays static plots and animations.

    Parameters
    ----------
    file : str or simulation.Simulation
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

    Returns
    -------

    """
    if "DISPLAY" not in os.environ.keys():
        print("Can't plot, DISPLAY not defined!")
        return False
    if show_static or show_animation or save_animation or save_static:
        if isinstance(file, simulation.Simulation):
            S = file
            S.postprocess()
        else:
            try:
                print(f"Loading simulation data from {file}")
                S = simulation.load_simulation(file)
            except:
                raise ValueError("Simulation file doesn't exist.")
        if show_animation or save_animation or snapshot_animation:
            # noinspection PyUnusedLocal
            # this needs name due to matplotlib.animation
            anim = animation_type(S, alpha, frames)
            if snapshot_animation:
                anim.snapshot_animation()
            if save_animation or show_animation:
                anim_object = anim.full_animation(save_animation)
        if save_static or show_static:
            filename = S.filename.replace(".hdf5", ".png") if save_static else None
            static = static_plots.static_plots_large(S, filename)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="hdf5 file name for storing data")
    parser.add_argument('-save', action='store_false')
    parser.add_argument('-lines', action='store_true')
    args = parser.parse_args()
    if args.filename[-5:] != ".hdf5":
        args.filename += ".hdf5"

    plots(args.filename, True, False, True, False)
