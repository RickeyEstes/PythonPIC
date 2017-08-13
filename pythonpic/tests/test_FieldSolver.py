# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ..classes import Simulation
from pythonpic.classes import PeriodicTestGrid, NonperiodicTestGrid
from pythonpic.classes import TestSpecies as Species
from ..visualization.time_snapshots import FieldPlot, CurrentPlot


@pytest.fixture(params=(64, 128, 256, 512))
def _NG(request):
    return request.param


@pytest.fixture(params=(1, 2 * np.pi, 10 * np.pi, 1000))
def _L(request):
    return request.param


@pytest.fixture(params=(1, 2 * np.pi, 10 * np.pi, 1000))
def _test_charge_density(request):
    return request.param

@pytest.fixture(params=(1, 2 * np.pi, 7.51))
def __t(request):
    return request.param

def test_PoissonSolver(_NG, _L):
    g = PeriodicTestGrid(1, _L, _NG)
    charge_density = (2 * np.pi / _L) ** 2 * np.sin(2 * g.x * np.pi / _L)
    field = np.zeros((_NG + 2, 3))
    field[1:-1, 0] = -2 * np.pi / _L * np.cos(2 * np.pi * g.x / _L)
    g.charge_density[:-1] = charge_density
    g.init_solve()

    def plots():
        fig, axes = plt.subplots(2)
        ax0, ax1 = axes
        ax0.plot(g.x, charge_density)
        ax0.set_title("Charge density")
        ax1.set_title("Field")
        ax1.plot(g.x, g.electric_field[1:-1], "r-", label="Fourier")
        ax1.plot(g.x, field, "g-", label="Analytic")
        for ax in axes:
            ax.grid()
            ax.legend()
        plt.show()
        return "test_PoissonSolver failed! calc/theory field ratio at 0: {}".format(g.electric_field[1] / field[0])

    assert np.allclose(g.electric_field, field), plots()


# def test_PoissonSolver_complex(debug=DEBUG):
#     L = 1
#     N = 32 * 2**5
#     epsilon_0 = 1
#     x, dx = np.linspace(0, L, N, retstep=True, endpoint=False)
#     anal_potential = lambda x: np.sin(x * 2 * np.pi) + 0.5 * \
#         np.sin(x * 6 * np.pi) + 0.1 * np.sin(x * 20 * np.pi)
#     anal_field = lambda x: -(2 * np.pi * np.cos(x * 2 * np.pi) + 3 * np.pi *
#                    np.cos(x * 6 * np.pi) + 20 * np.pi * 0.1 * np.cos(x * 20 * np.pi))
#     charge_density_anal = lambda x: ((2 * np.pi)**2 * np.sin(x * 2 * np.pi) + 18 * np.pi**2 * np.sin(
#         x * 6 * np.pi) + (20 * np.pi)**2 * 0.1 * np.sin(x * 20 * np.pi)) * epsilon_0
#
#     NG = 32
#     g = Frame(L, NG, epsilon_0)
#     # indices_in_denser_grid = np.searchsorted(x, g.x)
#     g.charge_density = charge_density_anal(g.x)
#     energy_fourier = g.init_solver_fourier()
#     energy_direct = 0.5 * (g.electric_field**2).sum() * g.dx
#     print("dx", dx, "fourier", energy_fourier, "direct", energy_direct, energy_fourier / energy_direct)
#
#     def plots():
#         fig, xspace = plt.subplots()
#         xspace.set_title(
#             r"Solving the Poisson equation $\Delta \psi = \rho / \epsilon_0$ via Fourier transform")
#         xspace.plot(g.x, g.charge_density, "ro--", label=r"$\rho$")
#         xspace.plot(x, charge_density_anal(x), "r-", lw=6, alpha=0.5, label=r"$\rho_a$")
#         xspace.plot(g.x, g.potential, "go--", label=r"$V$")
#         xspace.plot(x, anal_potential(x), "g-", lw=6, alpha=0.5, label=r"$V_a$")
#         xspace.plot(g.x, g.electric_field, "bo--", alpha=0.5, label=r"$E$")
#         EplotAnal, = xspace.plot(x, anal_field(x), "b-", lw=6, alpha=0.5, label=r"$E_a$")
#         xspace.set_xlim(0, L)
#         xspace.set_xlabel("$x$")
#         xspace.grid()
#         xspace.legend(loc='best')
#
#         fig2, fspace = plt.subplots()
#         fspace.plot(g.k_plot, g.energy_per_mode, "bo--", label=r"electric energy $\rho_F V_F^\dagger$")
#         fspace.set_xlabel("k")
#         fspace.set_ylabel("mode energy")
#         fspace.set_title("Fourier space")
#         fspace.grid()
#         fspace.legend(loc='best')
#         plt.show()
#         return "test_PoissonSolver_complex failed!"
#
#     energy_correct = np.isclose(energy_fourier, energy_direct)
#     field_correct = np.isclose(g.electric_field, anal_field(g.x)).all()
#     potential_correct = np.isclose(g.potential, anal_potential(g.x)).all()
#     assert field_correct and potential_correct and energy_correct, plots()

def test_PoissonSolver_energy_sine(_NG, ):
    _L = 1
    resolution_increase = _NG
    N = _NG * resolution_increase
    epsilon_0 = 1
    x, dx = np.linspace(0, _L, N, retstep=True, endpoint=False)
    anal_field = np.zeros((N, 3))
    anal_field[:, 0] = -(2 * np.pi * np.cos(x * 2 * np.pi / _L))

    charge_density_anal = ((2 * np.pi) ** 2 * np.sin(x * 2 * np.pi))

    g = PeriodicTestGrid(1, _L, _NG, epsilon_0)
    indices_in_denser_grid = np.searchsorted(x, g.x)
    g.charge_density[:-1] = charge_density_anal[indices_in_denser_grid]  # / resolution_increase

    g.init_solve()
    g.save_field_values(0)
    g.postprocess()
    energy_fourier = g.grid_energy_history[0]
    energy_direct = g.direct_energy_calculation()
    print("dx", dx, "fourier", energy_fourier, "direct", energy_direct, energy_fourier / energy_direct)

    def plots():
        fig, xspace = plt.subplots()
        xspace.set_title(
            r"Solving the Poisson equation $\Delta \psi = \rho / \epsilon_0$ via Fourier transform")
        xspace.plot(g.x, g.charge_density, "ro--", label=r"$\rho$")
        xspace.plot(x, charge_density_anal, "r-", lw=6, alpha=0.5, label=r"$\rho_a$")
        xspace.plot(g.x, g.electric_field, "bo--", alpha=0.5, label=r"$E$")
        xspace.plot(x, anal_field, "b-", lw=6, alpha=0.5, label=r"$E_a$")
        xspace.set_xlim(0, _L)
        xspace.set_xlabel("$x$")
        xspace.grid()
        xspace.legend(loc='best')

        fig2, fspace = plt.subplots()
        fspace.plot(g.k_plot, g.energy_per_mode, "bo--", label=r"electric energy $\rho_F V_F^\dagger$")
        fspace.set_xlabel("k")
        fspace.set_ylabel("mode energy")
        fspace.set_title("Fourier space")
        fspace.grid()
        fspace.legend(loc='best')
        plt.show()
        return "test_PoissonSolver_complex failed!"


    energy_correct = np.allclose(energy_fourier, energy_direct)
    assert energy_correct, plots()
    field_correct = np.allclose(g.electric_field[1:-1, 0], anal_field[indices_in_denser_grid][:, 0])
    assert field_correct, plots()


def test_PoissonSolver_sheets(_NG, _L, _test_charge_density=1):
    epsilon_0 = 1

    x, dx = np.linspace(0, _L, _NG, retstep=True, endpoint=False)
    charge_density = np.zeros_like(x)
    region1 = (_L * 1 / 8 < x) * (x < _L * 2 / 8)
    region2 = (_L * 5 / 8 < x) * (x < _L * 6 / 8)
    charge_density[region1] = _test_charge_density
    charge_density[region2] = -_test_charge_density
    g = PeriodicTestGrid(1, _L, _NG, epsilon_0)
    g.charge_density[:-1] = charge_density
    g.init_solve()

    def plots():
        fig, axes = plt.subplots(3)
        ax0, ax1 = axes
        ax0.plot(x, charge_density)
        ax0.set_title("Charge density")
        ax1.set_title("Field")
        ax1.plot(x, g.electric_field, "r-")
        for ax in axes:
            ax.grid()
            ax.legend()
        plt.show()
        return "test_PoissonSolver_sheets failed!"


    polynomial_coefficients = np.polyfit(x[region1], g.electric_field[1:-1, 0][region1], 1)
    first_bump_right = np.isclose(
        polynomial_coefficients[0], _test_charge_density, rtol=1e-2)

    assert first_bump_right, plots()
    polynomial_coefficients = np.polyfit(x[region2], g.electric_field[1:-1, 0][region2], 1)
    second_bump_right = np.isclose(
        polynomial_coefficients[0], -_test_charge_density, rtol=1e-2)
    assert second_bump_right, plots()


def test_PoissonSolver_ramp(_NG, _L):
    """ For a charge density rho = Ax + B
    d2phi/dx2 = -rho/epsilon_0
    set epsilon_0 to 1
    d2phi/dx2 = Ax
    phi must be of form
    phi = -Ax^3/6 + Bx^2 + Cx + D"""

    a = 1

    # noinspection PyArgumentEqualDefault
    g = PeriodicTestGrid(1, _L, _NG, epsilon_0=1)
    g.charge_density[:-1] = a * g.x
    g.init_solve()
    field = a * (g.x - _L / 2) ** 2 / 2

    def plots():
        fig, axes = plt.subplots(2)
        ax0, ax1 = axes
        ax0.plot(g.x, g.charge_density)
        ax0.set_title("Charge density")
        ax1.set_title("Field")
        ax1.plot(g.x, g.electric_field, "r-")
        ax1.plot(g.x, field, "g-")
        for ax in axes:
            ax.grid()
            ax.legend()
        plt.show()
        return "test_PoissonSolver_ramp failed!"


    polynomial_coefficients = np.polyfit(g.x, g.electric_field[1:-1, 0], 2)
    assert np.isclose(polynomial_coefficients[0], a / 2, rtol=1e-2), (polynomial_coefficients[0], a / 2, plots())

def test_BunemanSolver(__t, _NG, _L, _test_charge_density):
    g = NonperiodicTestGrid(__t, _L, _NG)
    charge_index = _NG // 2
    g.current_density_x[charge_index] = _test_charge_density
    g.solve()
    g.save_field_values(0)

    S = Simulation(g)
    pulled_field = g.electric_field[charge_index, 0]
    expected_field = - g.dt / g.epsilon_0 * _test_charge_density

    def plot():
        fig, (ax1, ax2) = plt.subplots(2)
        CurrentPlot(S, ax1, 0).update(0)
        FieldPlot(S, ax2, 0).update(0)
        plt.show()
    assert np.isclose(pulled_field, expected_field), plot()


def test_BunemanSolver_charge(__t, _NG, _L, _test_charge_density):
    g = NonperiodicTestGrid(__t, _L, _NG)
    v = 0.5
    g.current_density_x[1:-2] = v * _test_charge_density
    g.solve()
    g.save_field_values(0)
    S = Simulation(g).postprocess()


    def plot():
        fig, (ax1, ax2) = plt.subplots(2)
        CurrentPlot(S, ax1, 0).update(0)
        FieldPlot(S, ax2, 0).update(0)
        plt.show()
    assert np.allclose(g.electric_field[1:-1,0], -v * _test_charge_density * g.dt / g.epsilon_0), plot()


