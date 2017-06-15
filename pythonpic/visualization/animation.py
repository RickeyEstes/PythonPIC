"""Animates the simulation to show quantities that change over time"""
# coding=utf-8

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np

from .time_snapshots import FrequencyPlot, \
    PhasePlot, SpatialDistributionPlot, IterationCounter, \
    TripleFieldPlot, TripleCurrentPlot, CurrentPlot, FieldPlot, ChargeDistributionPlot, SpatialPerturbationDistributionPlot
from ..helper_functions import helpers


# formatter = matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False)

class Animation:
    def __init__(self,
                 S,
                 alpha=1,
                 frames="few"
                 ):

        """

        Creates an animation from `Plot`s.

        Parameters
        ----------
        S : Simulation
            Data source
        alpha : float
            Opacity value from 0 to 1, used in the phase plot.

        Returns
        -------
        figure or matplotlib animation
            Plot object, depending on `frame_to_draw`.
        """
        assert alpha <= 1, "alpha too large!"
        assert alpha >= 0, "alpha must be between 0 and 1!"
        self.S = S.postprocess()
        self.fig = plt.figure(figsize=(13, 10))
        self.alpha = alpha


        self.fig.suptitle(str(self.S), fontsize=12)
        self.fig.subplots_adjust(top=0.81, bottom=0.08, left=0.15, right=0.95,
                            wspace=.25, hspace=0.6)  # REFACTOR: remove particle windows if there are no particles


        if frames == "all":
            self.frames = np.arange(0, S.NT,
                                    dtype=int)
            self.frames_to_draw = self.frames
        elif frames == "few":
            self.frames = np.arange(0, S.NT,
                                    helpers.calculate_particle_iter_step(S.NT),
                                    dtype=int)
            self.frames_to_draw = self.frames[::30]
        elif isinstance(frames, list):
            self.frames_to_draw = frames
        else:
            raise ValueError("Incorrect frame_to_draw - must be 'animation' or number of iteration to draw.")

    def add_plots(self, plots):
        self.plots = plots

        self.updatable = []
        for plot in plots:
            for result in plot.return_animated():
                self.updatable.append(result)

    def animate(self, i, verbose=False):
        """draws the i-th frame of the simulation"""
        if self.S.considered_large or verbose:
            helpers.report_progress(i, self.S.grid.NT)
        for plot in self.plots:
            plot.update(i)
        return self.updatable

    def init(self):
        """initializes animation window for faster drawing"""
        for plot in self.plots:
            plot.animation_init()
        return self.updatable

    def full_animation(self, save=False, writer="ffmpeg"):
        """

        Parameters
        ----------
        save : bool
            Whether to save the simulation. This may take a long time.
        writer :

        Returns
        -------

        """
        print("Drawing full animation.")
        # noinspection PyTypeChecker
        animation_object = anim.FuncAnimation(self.fig, self.animate, interval=100,
                                              frames=self.frames,
                                              blit=True, init_func=self.init,
                                              fargs=(save,))
        if save:
            mpl_Writer = anim.writers[writer]
            mpl_writer = mpl_Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800) # TODO: does bitrate matter here

            helpers.make_sure_path_exists(self.S.filename)
            videofile_name = self.S.filename.replace(".hdf5", ".mp4")
            print(f"Saving animation to {videofile_name}")
            animation_object.save(videofile_name, writer=mpl_writer)#, extra_args=['-vcodec', 'libx264'])
            print(f"Saved animation to {videofile_name}")
        return animation_object

    def snapshot_animation(self):
        print("Drawing animation as snapshots.")
        for i in self.frames_to_draw:
            self.animate(i)
            helpers.make_sure_path_exists(self.S.filename)
            file_name = self.S.filename.replace(".hdf5", f"_{i:06}.png")
            print(f"Saving iteration {i} to {file_name}")
            self.fig.savefig(file_name)
        return self.fig

from matplotlib import gridspec
class FullAnimation(Animation):
    def __init__(self, S, alpha=1, frames="few"):
        super().__init__(S, alpha=alpha, frames=frames)
        gs = gridspec.GridSpec(4, 3, )
        phase_axes = [plt.subplot(gs[0, i]) for i in range(3)]
        current_axes = [plt.subplot(gs[1, i]) for i in range(3)]
        field_axes = [plt.subplot(gs[2, i]) for i in range(3)]
        bonus_row = [plt.subplot(gs[3, i]) for i in range(3)]
        density_axis, density_perturbation_axis, charge_axes  = bonus_row

        if any([species.individual_diagnostics for species in S.list_species]):
            phase_plot_x = PhasePlot(self.S, phase_axes[0], "x", "v_x", self.alpha)
            phase_plot_y = PhasePlot(self.S, phase_axes[1], "x", "v_y", self.alpha)
            phase_plot_z = PhasePlot(self.S, phase_axes[2], "x", "v_z", self.alpha)
            plots = [phase_plot_x, phase_plot_y, phase_plot_z]
        else:
            plots = []
        charge_plot = ChargeDistributionPlot(self.S, charge_axes)
        density_plot = SpatialDistributionPlot(self.S, density_axis)
        iteration = IterationCounter(self.S, charge_axes)
        current_plots = TripleCurrentPlot(self.S, current_axes)
        field_plots = TripleFieldPlot(self.S, field_axes)
        density_perturbation_plot = SpatialPerturbationDistributionPlot(S, density_perturbation_axis)

        plots += [
                 charge_plot,
                 density_plot,
                 iteration,
                 current_plots,
                 density_perturbation_plot,
                 field_plots]
        super().add_plots(plots)

class FastAnimation(Animation):
    def __init__(self, S, alpha=1, frames="few"):
        super().__init__(S, alpha=alpha, frames=frames)
        density_axis = self.fig.add_subplot(421)
        current_axes = [self.fig.add_subplot(423 + 2 * i) for i in range(3)]
        field_axes = [self.fig.add_subplot(424 + 2 * i) for i in range(2)]
        charge_axis = self.fig.add_subplot(428)
        density_perturbation_axis = self.fig.add_subplot(422)

        charge_plot = ChargeDistributionPlot(self.S, charge_axis)
        density_perturbation_plot = SpatialPerturbationDistributionPlot(S, density_perturbation_axis)
        density_plot = SpatialDistributionPlot(self.S, density_axis)
        iteration = IterationCounter(self.S, charge_axis)
        current_plots = TripleCurrentPlot(self.S, current_axes)
        field_plots = TripleFieldPlot(self.S, field_axes)

        plots = [
                 charge_plot,
                 density_plot,
                 iteration,
                 current_plots,
                 density_perturbation_plot,
                 field_plots]
        super().add_plots(plots)

class OneDimAnimation(Animation):
    def __init__(self, S, alpha=0.6, frames="few"):
        super().__init__(S, alpha=alpha, frames=frames)
        density_axis = self.fig.add_subplot(321)
        charge_axis = self.fig.add_subplot(323)

        current_axis = self.fig.add_subplot(324)
        field_axis = self.fig.add_subplot(326)
        phase_axes_x = self.fig.add_subplot(322)
        freq_axes = self.fig.add_subplot(325)

        if any([species.individual_diagnostics for species in S.list_species]):
            phase_plot_x = PhasePlot(self.S, phase_axes_x, "x", "v_x", self.alpha)
            plots = [phase_plot_x]
        else:
            plots = []
        freq_plot = FrequencyPlot(self.S, freq_axes)
        density_plot = SpatialDistributionPlot(self.S, density_axis)
        charge_plot = ChargeDistributionPlot(self.S, charge_axis)
        iteration = IterationCounter(self.S, freq_axes)
        current_plot = CurrentPlot(self.S, current_axis, 0)
        field_plot = FieldPlot(self.S, field_axis, 0)

        plots += [
                 freq_plot,
                 density_plot,
                 charge_plot,
                 iteration,
                 current_plot,
                 field_plot]
        super().add_plots(plots)

class ParticleDensityAnimation(Animation):
    def __init__(self, S, alpha, frames="few"):
        super().__init__(S, alpha=alpha, frames=frames)
        density_axis = self.fig.add_subplot(221)
        charge_x_axis = self.fig.add_subplot(222)
        charge_y_axis = self.fig.add_subplot(223)
        charge_z_axis = self.fig.add_subplot(224)
        current_plots = TripleCurrentPlot(self.S, [charge_x_axis, charge_y_axis, charge_z_axis])
        density_plot = SpatialDistributionPlot(self.S, density_axis)
        iteration = IterationCounter(self.S, charge_x_axis)


        plots = [
            density_plot,
            current_plots,
            iteration]
        super().add_plots(plots)



