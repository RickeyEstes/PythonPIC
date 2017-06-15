# coding=utf-8
import itertools

import numpy as np
import matplotlib.pyplot as plt

from ..helper_functions.helpers import is_this_saved_iteration, convert_global_to_particle_iter, colors, directions


class Plot:
    """
    A plot for visualization. Mainly an abstract class for overloading with interesting kinds of diagnostics.

    Parameters
    ----------
    S : Simulation
        A `Simulation` object to pull data from.
    ax : matplotlib axis
        An axis to draw on
    """

    def __init__(self, S, ax):
        self.S = S
        if isinstance(ax, str):
            fig, self.ax = plt.subplots()
        else:
            self.ax = ax
        self.plots = []

        L = S.grid.L
        self.ax.set_xlim(0, S.grid.L)
        self.ax.set_xlabel(rf"Position $x$ (L={L:.3e} m)")
        self.ax.grid()
        xticks = np.linspace(0, L, 7)
        self.ax.set_xticks(xticks)
        self.ax.xaxis.set_ticklabels([f"{x/L:.1f}L" for x in xticks])
        self.ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True,
                                 useOffset=False)  # TODO axis=both?
        # self.ax.yaxis.set_label_position("right")

    def animation_init(self):
        """
        Zeroes out all data in all lines of the plot. Useful for Animation.
        """
        for plot in self.plots:
            plot.set_data([], [])

    def update(self, i):
        """
        Updates the plot with information from a particular iteration of the simulation.

        Parameters
        ----------
        i : int
            Iteration of the simulation
        """
        pass

    def return_animated(self):
        """
        Returns an iterable of all items that have changed. Useful for Animation
        """
        return self.plots


class FrequencyPlot(Plot):
    """
    Plots the spatial Fourier transform of field energy versus wave number.
    """  # REFACTOR move the fourier analysis to PostProcessedGrid; describe the math here as well as there

    def __init__(self, S, ax):
        super().__init__(S, ax)
        self.plots.append(self.ax.plot([], [], "o-", label="energy per mode")[0])
        self.ax.set_xlabel(r"Wavevector mode $k$")
        self.ax.set_ylabel(r"Energy $E$")
        max_interesting = S.grid.k_plot[...].max() * 0.3
        # self.indices = S.grid.k_plot <  max_interesting
        self.indices = np.ones_like(S.grid.k_plot, dtype=bool)
        interesting_x = S.grid.k_plot[self.indices]
        self.ax.set_xticks(interesting_x)
        self.ax.xaxis.set_ticklabels(np.arange(len(interesting_x)))
        self.ax.set_xlim(interesting_x.min(), interesting_x.max())
        self.ax.set_ylim(0, S.grid.longitudinal_energy_per_mode_history[...].max())

    def update(self, i):
        # import ipdb; ipdb.set_trace()
        self.plots[0].set_data(self.S.grid.k_plot[self.indices],
                               self.S.grid.longitudinal_energy_per_mode_history[i][self.indices])


def phaseplot_values(species):
    """
    A convenience function to get a dictionary of values, to allow generalization of the PhasePlot class.

    The keys you can pull for phase plots are `x`, `v_x`, `v_y` and `v_z`.

    Parameters
    ----------
    species : Species
        A species to draw data from

    Returns
    -------
    A dictionary of phase plot values.
    """
    return {"x":                           species.position_history,
            "v_x": species.velocity_history[:, :, 0],
            "v_y": species.velocity_history[:, :, 1],
            "v_z": species.velocity_history[:, :, 2],
            }


class PhasePlot(Plot):
    """
    Draws a phase plot.

    The keys you can pull for phase plots are `x`, `v_x`, `v_y` and `v_z`.

    Parameters
    ----------
    v1, v2 : str
        keys for the phase plot.
    alpha : float
        An opacity value between 0 and 1. Useful for neat phase plots displaying density.
    """

    def __init__(self, S, ax, v1, v2, alpha):
        super().__init__(S, ax)
        self.x = [phaseplot_values(species)[v1] for species in S.list_species]
        self.y = [phaseplot_values(species)[v2] for species in S.list_species]
        if len(self.y):
            maxys = max([np.max(np.abs(y)) for y in self.y])
            self.ax.set_ylim(-maxys, maxys)
        for i, species in enumerate(S.list_species):
            self.plots.append(self.ax.plot([], [], colors[i] + ".", alpha=alpha)[0])
        # self.ax.yaxis.set_label_position("right")
        self.ax.set_xlabel(rf"${v1}$")
        self.ax.set_ylabel(rf"${v2}$")

    def update(self, i):
        for plot, species, x, y in zip(self.plots, self.S.list_species, self.x, self.y):
            if is_this_saved_iteration(i, species.save_every_n_iterations):
                index = convert_global_to_particle_iter(i, species.save_every_n_iterations)
                alive = species.N_alive_history[index] +1
                # print(y[index, species.alive_history[index]]) #TODO: get alive history to work here!
                plot.set_data(x[index, :alive],  # , species.alive_history[index]],
                              y[index, :alive])  # , species.alive_history[index]])


class SpatialDistributionPlot(Plot):
    """
    Draws particle density on the grid.
    """

    def __init__(self, S, ax):
        super().__init__(S, ax)
        ax.set_ylabel(f"Particle density $n$")
        for species in S.list_species:
            self.plots.append(self.ax.plot([], [], "-", label=species.name)[0])
        if len(S.list_species):
            self.ax.set_ylim(0, 1.2*max([species.density_history[...].max() for species in S.list_species]))
            self.ax.legend(loc='best')

    def update(self, i):
        for species, plot in zip(self.S.list_species, self.plots):
            plot.set_data(self.S.grid.x, species.density_history[i])

class SpatialPerturbationDistributionPlot(SpatialDistributionPlot):
    def __init__(self, S, ax):
        super().__init__(S, ax)
        self.ax.set_ylabel(r"$\Delta n = n - n(t=0)$")
        self.y = [species.density_history - species.density_history[0] for species in S.list_species]
        if len(S.list_species):
            self.ax.set_ylim(min([1.2 * y.min() for y in self.y]),max([1.2 * y.max() for y in self.y]))
            self.ax.legend(loc='best')

    def update(self, i):
        for species, plot, y in zip(self.S.list_species, self.plots, self.y):
            plot.set_data(self.S.grid.x, y[i])

class ChargeDistributionPlot(Plot):
    """
    Draws charge density from the grid.
    """

    def __init__(self, S, ax, check_poisson=False):
        super().__init__(S, ax)
        self.plots.append(self.ax.plot([], [], "-", alpha=0.8, label="charge")[0])
        self.ax.set_ylabel(f"Charge density $\\rho$")
        mincharge = np.min(S.grid.charge_density_history)
        maxcharge = np.max(S.grid.charge_density_history)
        self.ax.set_ylim(mincharge, maxcharge)
        self.check_poisson = check_poisson
        if check_poisson:
            self.plots.append(self.ax.plot([], [], "-", alpha=0.8, label=r"$\varepsilon_0 \partial E/ \partial x$")[0])
            self.ax.legend(loc='lower left')

    def update(self, i):
        self.plots[0].set_data(self.S.grid.x, self.S.grid.charge_density_history[i, :])
        if self.check_poisson:
            self.plots[1].set_data(self.S.grid.x, self.S.grid.check_on_charge[i])

class Histogram(Plot):
    """
    Draws a histogram of a given value from the phase plot dataset.

    The keys you can pull for phase plots are `x`, `v_x`, `v_y` and `v_z`.

    Parameters
    ----------
    v1 : str
        A key to phase plot values.
    n_bins: int
        Number of bins to draw.
    """

    def __init__(self, S, ax, v1: str, n_bins: int = 50):
        super().__init__(S, ax)
        self.bin_arrays = []
        self.values = [phaseplot_values(species)[v1] for species in S.list_species]
        if len(self.values):
            maxxs = max([np.max(np.abs(v)) for v in self.values])
            self.ax.set_xlim(-maxxs, maxxs)
        for i, s, v in zip(range(len(S.list_species)), S.list_species, self.values):
            bin_array = np.linspace(v.min(), v.max(), n_bins)
            self.bin_arrays.append(bin_array)
            self.plots.append(
                self.ax.plot(*calculate_histogram_data(v[0], bin_array), colors[i])[0])
        self.ax.set_xlabel(rf"${v1}$")
        self.ax.set_ylabel(r"Number of particles")

        if len(self.bin_arrays):
            self.ax.set_xlim(min([bin_array.min() for bin_array in self.bin_arrays]),
                             max([bin_array.max() for bin_array in self.bin_arrays]))

    def update(self, i):
        for species, histogram, bin_array, v in zip(self.S.list_species, self.plots, self.bin_arrays, self.values):
            index = convert_global_to_particle_iter(i, species.save_every_n_iterations)
            alive = species.N_alive_history[index] +1
            histogram.set_data(*calculate_histogram_data(v[index, :alive], bin_array))


def calculate_histogram_data(arr, bins):
    """
    Calculates histogram values, normalized to the number of particles.

    Parameters
    ----------
    arr : ndarray
        Values of a particle property, for example, velocity
    bins : ndarray
        Bin edges for the histogram.
    Returns
    -------
    bin_center : ndarray
        Centers of histogram bars (the x array for plotting)
    bin_height : ndarray
        Heights of histogram bars (the y array for plotting)
    """
    bin_height, bin_edge = np.histogram(arr, bins=bins)  # OPTIMIZE
    bin_center = (bin_edge[:-1] + bin_edge[1:]) * 0.5
    return bin_center, bin_height


class IterationCounter:
    """
    A little widget inserted on an axis, displaying the iteration number and current simulation time.
    """

    def __init__(self, S, ax):
        self.S = S
        self.ax = ax
        self.counter = ax.text(0.1, 0.9, 'i=x', horizontalalignment='left',
                               verticalalignment='center', transform=ax.transAxes)

    def animation_init(self):
        self.counter.set_text("Iteration: \nTime: ")

    def update(self, i):
        self.counter.set_text(f"Iteration: {i}/{self.S.NT}\nTime: {i*self.S.dt:.3g}/{self.S.NT*self.S.dt:.3g}")

    def return_animated(self):
        return [self.counter]


class FieldPlot(Plot):
    """
    Draws electric and magnetic fields from the grid in a given direction

    Parameters
    ----------
    j : int
        Direction as Cartesian index number. 0: x, 1: y, 2: z
    """

    def __init__(self, S, ax, j):
        super().__init__(S, ax)
        self.j = j
        self.plots.append(self.ax.plot([], [], "-", label=f"$E_{directions[j]}$")[0])
        self.ax.set_ylabel(r"Fields $E$, $B$")
        max_e = np.max(np.abs(S.grid.electric_field_history[:, :, j]))
        if j != 0:
            self.plots.append(self.ax.plot([], [], "-", label=f"$B_{directions[j]}$")[0])
            max_b = np.max(np.abs(S.grid.magnetic_field_history[:, :, j]))
            maxfield = max([max_e, max_b])
        else:
            maxfield = max_e
        print(f"For direction {directions[j]}, maxfield is {maxfield:.2e}")
        self.ax.set_ylim(-maxfield, maxfield)
        self.ax.legend(loc='upper right')

    def update(self, i):
        self.plots[0].set_data(self.S.grid.x, self.S.grid.electric_field_history[i, :, self.j])
        if self.j != 0:
            self.plots[1].set_data(self.S.grid.x, self.S.grid.magnetic_field_history[i, :, self.j])


class CurrentPlot(Plot):
    """
    Draws currents from the grid in a given direction.

    Parameters
    ----------
    j : int
        Direction as Cartesian index number. 0: x, 1: y, 2: z
    """

    def __init__(self, S, ax, j):
        super().__init__(S, ax)
        self.j = j
        x = S.grid.x_current if j == 0 else S.grid.x
        self.plots.append(self.ax.plot(x, S.grid.current_density_history[0, :, j], "-",
                                  alpha=0.9,
                                  label=fr"$j_{directions[j]}$")[0])
        self.ax.set_ylabel(f"Current density $j_{directions[j]}$")
        self.ax.tick_params('y')
        self.ax.legend(loc='lower left')

        current = S.grid.current_density_history[:, :, j]
        # mean = current.mean()
        # std = 3*current.std()
        #
        # mincurrent = mean - std
        # maxcurrent = mean + std
        mincurrent = current.min()
        maxcurrent = current.max()
        try:
            ax.set_ylim(mincurrent, maxcurrent)
        except ValueError as E:
            print(f"Error on setting current limits in {j}: {E}")

    def update(self, i):
        self.plots[0].set_data(self.S.grid.x, self.S.grid.current_density_history[i, :, self.j])


class PlotSet:
    """
    A single object representing a few different plots on different axes.
    Useful for plotting sets of directional values (fields, currents).

    Parameters
    ----------
    axes : list
        List of axes to use.
    list_plots :
        List of `Plot`s to update and return.
    """

    def __init__(self, axes, list_plots):
        self.axes = axes
        self.list_plots = list_plots

    def update(self, i):
        for plot in self.list_plots:
            plot.update(i)

    def animation_init(self):
        for plot in self.list_plots:
            plot.animation_init()

    def return_animated(self):
        return list(itertools.chain.from_iterable(plot.return_animated() for plot in self.list_plots))


class TripleFieldPlot(PlotSet):
    """
    Draws electric and magnetic field plots on the grid on a given list of axes.
    Parameters
    ----------
    S : Simulation
        Simulation to pull data from.
    axes : list
        List of matplotlib axes.
    """

    def __init__(self, S, axes: list):
        assert len(axes) <= 3, "Too many axes, we ran out of directions!"
        plots = [FieldPlot(S, ax, j) for j, ax in enumerate(axes)]
        super().__init__(axes, plots)


class TripleCurrentPlot(PlotSet):
    """
    Draws currents on the grid on a given list of axes.
    Parameters
    ----------
    S : Simulation
        Simulation to pull data from.
    axes : list
        List of matplotlib axes.
    """

    def __init__(self, S, axes: list):
        assert len(axes) <= 3, "Too many axes, we ran out of directions!"
        plots = [CurrentPlot(S, ax, j) for j, ax in enumerate(axes)]
        super().__init__(axes, plots)
