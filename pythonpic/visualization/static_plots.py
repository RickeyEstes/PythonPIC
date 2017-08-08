# coding=utf-8
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from ..helper_functions.helpers import calculate_particle_snapshots, colors, directions

class TimePlot:
    def __init__(self, S, ax):
        self.S = S
        if isinstance(ax, str):
            fig, self.ax = plt.subplots()
        else:
            self.ax = ax
        self.plots = []
        self.t = self.S.grid.t
        self.ax.set_xlim(0, self.grid.T)
        self.ax.set_xlabel(rf"Time $t$ (T={T:.3e} s)")
        self.ax.grid()
        tticks = np.linspace(0, self.grid.T, 7)
        self.ax.set_xticks(tticks)
        self.ax.xaxis.set_ticklabels([f"{T/L:.1f}L" for t in tticks])
        self.ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True,
                                 useOffset=False)  # TODO axis=both?

def static_plot_window(S, N, M):
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(N, M)
    axes = [[fig.add_subplot(gs[n, m]) for m in range(M)] for n in range(N)]
    fig.suptitle(str(S), fontsize=12)

    gs.update(left=0.075, right=0.95, bottom=0.075, top=0.8, hspace=0.45, wspace=0.25)  # , wspace=0.05, hspace=0.05
    return fig, axes


# REFACTOR: turn these into classes like in Animation
def ESE_time_plots(S, axis):
    data = S.grid.longitudinal_energy_per_mode_history
    # wavelengths = 2 * np.pi / S.grid.k_plot

    top_values = data.max(axis=0)
    sorted_indices = np.argsort(top_values)
    # weights = (data ** 2).sum(axis=0) / (data ** 2).sum()
    #
    # # noinspection PyUnusedLocal
    # max_mode = weights.argmax()
    # # TODO: max_index = data[:, max_mode].argmax()

    t = np.arange(S.NT) * S.dt
    for i in range(6):
        axis.plot(t, data[:, i], label=f"Mode {i}", alpha=0.8)
    for i in range(6, data.shape[1]):
        axis.plot(t, data[:, i], alpha=0.5)
    # axis.annotate(f"Mode {max_mode}",
    #               xy=(t[max_index], data[max_index, max_mode]),
    #               arrowprops=dict(facecolor='black', shrink=0.05),
    #               xytext=(t.mean(), data.max()/2))

    axis.legend(loc='best')
    axis.grid()
    axis.set_xlabel(f"Time [s] [dt: {S.dt:.2e}]")
    axis.set_ylabel("Energy [J/m^2]")
    axis.set_xlim(0, S.NT * S.dt)
    axis.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True, useOffset=False)
    axis.set_title("Energy per spatial longitudinal mode versus time")


def temperature_time_plot(S, axis, twinaxis=True):
    from matplotlib.font_manager import FontProperties
    fontP = FontProperties()
    fontP.set_size('small')

    for species in S.list_species:
        t = S.t
        meanv = species.velocity_mean_history[...]
        meanv2 = species.velocity_squared_mean_history[...]
        temperature = meanv2 - meanv**2
        temperature_t = temperature.sum(axis=1)
        axis.plot(t, temperature_t, label=species.name)
    axis.legend(loc='best', prop=fontP)
    axis.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True, useOffset=False)
    axis.grid()
    axis.set_xlabel(r"Time $t$")
    axis.set_xlim(0, S.NT * S.dt)
    axis.set_ylabel(r"Temperature ($\bar{v^2} - \bar{v}^2$) [$(\frac{m}{s})^2$]")


def energy_time_plots(S, axis, biaxial = False):
    if biaxial:
        twin = axis.twinx()
    else:
        twin = axis
    for species in S.list_species:
        axis.plot(S.t, species.kinetic_energy_history, "--",
                  label="Kin.: {}".format(species.name))
    axis.plot(np.arange(S.NT) * S.dt, S.grid.longitudinal_energy_history, "-", label="Long. E.", alpha=1)
    axis.plot(np.arange(S.NT) * S.dt, S.grid.perpendicular_energy_history, "-", label="Perp. E.", alpha=1)
    axis.grid()
    axis.set_xlabel(r"Time $t$")
    axis.set_xlim(0, S.NT * S.dt)
    axis.set_ylabel(r"Energy $E$ [$J/m^2$]")
    axis.legend(loc='best')
    if biaxial:
        twin.legend(loc='lower right')
        twin.set_ylabel("Kinetic energy")
    axis.set_title("Energy evolution")
    axis.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True, useOffset=False)

def alternate_energy_time_plots(S, axis, biaxial = False):
    if biaxial:
        twin = axis.twinx()
    else:
        twin = axis
    for species in S.list_species:
        twin.plot(S.t, species.kinetic_energy_history, "--",
                  label="Kin.: {}".format(species.name))
    axis.plot(np.arange(S.NT) * S.dt, S.grid.energy_via_bc_history, "-", label="Energy via Poynting", alpha=0.7)
    axis.plot(np.arange(S.NT) * S.dt, S.grid.entering_energy_via_bc_history, "-", label="Entering energy via Poynting", alpha=0.7)
    axis.plot(np.arange(S.NT) * S.dt, S.grid.longitudinal_energy_history, "-", label="Long. E.", alpha=1)
    axis.plot(np.arange(S.NT) * S.dt, S.grid.perpendicular_energy_history, "-", label="Perp. E.", alpha=1)
    # twin.plot(np.arange(S.NT) * S.dt, S.grid.laser_energy_history, "m-", label="Laser E.")
    axis.grid()
    axis.set_xlabel(r"Time $t$")
    axis.set_xlim(0, S.NT * S.dt)
    axis.set_ylabel(r"Energy $E$ [$J/m^2$]")
    axis.legend(loc='best')
    if biaxial:
        twin.legend(loc='lower right')
        twin.set_ylabel("Kinetic energy")
    axis.set_title("Energy evolution - alternate")
    axis.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True, useOffset=False)
def electrostatic_energy_time_plots(S, axis):
    for species in S.list_species:
        axis.plot(S.t, species.kinetic_energy_history, "-",
                  label="Kin.: {}".format(species.name))
    axis.plot(np.arange(S.NT) * S.dt, S.grid.longitudinal_energy_history, "-", label="Long. E.", alpha=0.7)
    axis.plot(np.arange(S.NT) * S.dt, S.total_energy_history, "--", label="Total E.", alpha=0.7)
    axis.grid()
    axis.set_xlabel(r"Time $t$")
    axis.set_xlim(0, S.NT * S.dt)
    axis.set_ylabel(r"Energy $E$ [$J/m^2$]")
    axis.legend(loc='best')
    axis.set_title("Energy evolution")
    axis.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True, useOffset=False)

def alive_time_plots(S, axis):
    for species in S.list_species:
        axis.plot(S.t, species.N_alive_history, "-", label=species.name)
    axis.grid()
    axis.set_xlabel(r"Time $t$")
    axis.set_xlim(0, S.NT * S.dt)
    axis.set_ylabel(r"N of alive particles")
    axis.legend(loc='best')
    axis.set_title("Particle lifetime")
    axis.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True, useOffset=False)


def velocity_distribution_plots(S, axis, i=0):
    for species in S.list_species:
        index = i // species.save_every_n_iterations
        axis.hist(species.velocity_history[index, :, 0],
                  bins=50, alpha=0.5, normed=True,
                  label=f"{species.name} ({species.N_alive_history[i]} alive)")
    axis.set_title("Velocity distribution at iteration %d" % i)
    axis.grid()
    if len(S.list_species) > 1:
        axis.legend(loc='best')
    axis.set_xlabel(r"Velocity $v$")
    axis.set_ylabel(r"fraction of superparticles out")
    axis.ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True, useOffset=False)

def phase_trajectories(S, axis, all=False):
    for species in S.list_species:
        if all:
            x = species.position_history[:, :]
            y = species.velocity_history[:, :, 0]
            axis.set_title("Phase space plot")
        else:
            i = int(species.N / 2)
            x = species.position_history[:, i]
            y = species.velocity_history[:, i, 0]
            axis.set_title("Phase space plot for particle {}".format(i))
        axis.plot(x, y, ".", label=species.name)
    axis.set_xlim(0, S.grid.L)
    if len(S.list_species) > 1:
        axis.legend(loc='best')
    axis.set_xlabel(r"Position $x$ [m]")
    axis.set_ylabel(r"Velocity $v_x$ [m/s]")
    axis.grid()
    axis.ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True, useOffset=False)

def position_trajectories(S, axis, species_index, n_particle = "half"):
    species = S.list_species[species_index]
    if n_particle == "half":
        n_particle = int(species.N/2)
    axis.set_title("Phase space plot")
    y = species.position_history[:, n_particle]
    x = species.save_every_n_iterations * S.dt * np.arange(y.size)
    axis.set_title(f"Trajectory for particle {n_particle} / {species.N}")
    axis.plot(x, y, "o--", label=species.name)
    axis.set_xlim(0, S.grid.L)
    if len(S.list_species) > 1:
        axis.legend(loc='best')
    axis.set_xlabel(r"Time $t$ [s]")
    axis.set_ylabel(r"Position $x$ [m]")
    axis.grid()
    axis.ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True, useOffset=False)

def velocity_trajectories(S, axis, species_index, n_particle = "half", direction=0):
    species = S.list_species[species_index]
    if n_particle == "half":
        n_particle = int(species.N/2)
    axis.set_title("Phase space plot")
    y = species.velocity_history[:, n_particle, direction]
    x = species.save_every_n_iterations * S.dt * np.arange(y.size)
    axis.set_title(f"Velocity history for particle {n_particle} / {species.N}")
    axis.plot(x, y, "o--", label=species.name)
    axis.set_xlim(0, S.grid.L)
    if len(S.list_species) > 1:
        axis.legend(loc='best')
    axis.set_xlabel(r"Time $t$ [s]")
    axis.set_ylabel(r"Velocity [m/s]")
    axis.grid()
    axis.ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True, useOffset=False)

def velocity_time_plots(S, axis):
    for s in S.list_species:
        # t = np.arange(calculate_particle_snapshots(S.NT), dtype=int) * S.dt * s.save_every_n_iterations
        for i in range(3):
            mean = s.velocity_mean_history[:, i]
            std = s.velocity_std_history[:, i]
            axis.plot(S.grid.t, mean, "-", color=colors[i], label=f"{s.name} $v_{directions[i]}$", alpha=1)
            axis.fill_between(S.grid.t, mean - std, mean + std, color=colors[i], alpha=0.3)
    axis.set_xlabel(r"Time $t$ [s]")
    axis.set_ylabel(r"Velocity $v$ [m/s]")
    if len(S.list_species) > 1:
        axis.legend(loc='best')
    axis.grid()
    axis.ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True, useOffset=False)

def directional_velocity_time_plots(S, axis, j):
    for i, s in enumerate(S.list_species):
        # t = np.arange(calculate_particle_snapshots(S.NT), dtype=int) * S.dt * s.save_every_n_iterations
        mean = s.velocity_mean_history[:, j]
        std = s.velocity_std_history[:, j]
        axis.plot(S.grid.t, mean, "-", color=colors[i], label=f"{s.name} $v_{directions[j]}$", alpha=1)
        axis.fill_between(S.grid.t, mean - std, mean + std, color=colors[i], alpha=0.3)
    axis.set_xlabel(r"Time $t$")
    axis.set_ylabel(r"Avg. vel. $<v> \pm 1 $ std [m/s]")
    if len(S.list_species) > 1:
        axis.legend(loc='best')
    axis.grid()
    axis.ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True, useOffset=False)

def total_velocity_time_plots(S, axis):
    for i, s in enumerate(S.list_species):
        # t = np.arange(calculate_particle_snapshots(S.NT), dtype=int) * S.dt * s.save_every_n_iterations
        mean = np.sqrt(np.sum(s.velocity_mean_history[...]**2, axis=1))
        # std = np.sqrt(np.sum(s.velocity_std_history[...]**2, axis=1))
        axis.plot(S.grid.t, mean, "-", color=colors[i], label=f"{s.name} $|v|$", alpha=1)
        # axis.fill_between(S.grid.t, mean - std, mean + std, color=colors[i], alpha=0.3)
    axis.set_xlabel(r"Time $t$")
    axis.set_ylabel(r"Avg. vel. $<|v|>$ [m/s]")
    if len(S.list_species) > 1:
        axis.legend(loc='best')
    axis.grid()
    axis.ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True, useOffset=False)

def static_plots(S, filename=None):
    print(S.grid)
    if filename and not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    time_fig, axes = static_plot_window(S, 3, 2)

    temperature_time_plot(S, axes[1][0])
    energy_time_plots(S, axes[2][0])
    total_velocity_time_plots(S, axes[0][0])
    # energy_time_plots(S, axes[0][0], biaxial=True)
    for i in range(2):
        directional_velocity_time_plots(S, axes[i][1], i)
        axes[i][1].yaxis.tick_right()
        axes[i][1].yaxis.set_label_position("right")

    alternate_energy_time_plots(S, axes[2][1])
    # axes[2][1].yaxis.tick_right()
    # axes[2][1].yaxis.set_label_position("right")

    if filename:
        time_fig.savefig(filename)
    return time_fig

def electrostatic_static_plots(S, filename=None):
    if filename and not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    time_fig, axes = static_plot_window(S, 3, 2)

    temperature_time_plot(S, axes[1][0])
    electrostatic_energy_time_plots(S, axes[2][0])
    total_velocity_time_plots(S, axes[0][0])
    # energy_time_plots(S, axes[0][0], biaxial=True)
    directional_velocity_time_plots(S, axes[0][1], 0)

    position_trajectories(S, axes[1][1], 0)
    velocity_trajectories(S, axes[2][1], 0)

    # for i in range(3):
    #     axes[i][1].yaxis.tick_right()
    #     axes[i][1].yaxis.set_label_position("right")

    if filename:
        time_fig.savefig(filename)
    return time_fig
def static_plots_large(S, filename=None):
    if filename and not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    time_fig, axes = static_plot_window(S, 3, 2)

    temperature_time_plot(S, axes[1][0])
    energy_time_plots(S, axes[2][0])
    total_velocity_time_plots(S, axes[0][0])
    # energy_time_plots(S, axes[0][0], biaxial=True)
    for i in range(3):
        directional_velocity_time_plots(S, axes[i][1], i)
        axes[i][1].yaxis.tick_right()
        axes[i][1].yaxis.set_label_position("right")


    if filename:
        time_fig.savefig(filename)
    return time_fig

def publication_plots(S, filename, list_plots):
    fig, axes = plt.subplots(len(list_plots))
    if type(axes) is not list:
        axes = [axes]
    for plot, ax in zip(list_plots, axes):
        plot(S, ax)

    fig.savefig(filename)
    return fig
