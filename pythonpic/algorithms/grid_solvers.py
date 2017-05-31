"""Numerical algorithms for the grid - interpolation to and from the grid, PDE solvers"""
# coding=utf-8
import numpy as np
from scipy import fftpack as fft


def PoissonLongitudinalSolver(rho, k, NG, epsilon_0=1, neutralize=True):
    """solves the Poisson equation spectrally, via FFT

    the Poisson equation can be written either as
    (in position space)
    $$\nabla \cdot E = \rho/\epsilon_0$$
    $$\nabla^2 V = -\rho/\epsilon_0$$

    Assuming that all functions in fourier space can be represented as
    $$\exp{i(kx - \omega t)}$$
    It is easy to see that upon Fourier transformation $\nabla \to ik$, so

    (in fourier space)
    $$E = \rho /(ik \epsilon_0)$$
    $$V = \rho / (-k^2 \epsilon_0)$$

    Calculate that, fourier transform back to position space
    and both the field and potential pop out easily

    The conceptually problematic part is getting the $k$ wave vector right
    # DOCUMENTATION: finish this description
    """

    rho_F = fft.fft(rho) # OPTIMIZE check if it's possible to use rfft here
    if neutralize:
        rho_F[0] = 0
    field_F = rho_F / (1j * k * epsilon_0)
    potential_F = field_F / (-1j * k * epsilon_0)
    field = fft.ifft(field_F).real
    return field


def BunemanTransversalSolver(electric_field, magnetic_field, current_yz, dt, c, epsilon_0):
    # dt = dx/c
    Fplus = 0.5 * (electric_field[:, 0] + c * magnetic_field[:, 1])
    Fminus = 0.5 * (electric_field[:, 0] - c * magnetic_field[:, 1])
    Gplus = 0.5 * (electric_field[:, 1] + c * magnetic_field[:, 0])
    Gminus = 0.5 * (electric_field[:, 1] - c * magnetic_field[:, 0])

    Fplus[1:] = Fplus[:-1] - 0.5 * dt * (current_yz[2:-1, 0]) / epsilon_0
    Fminus[:-1] = Fminus[1:] - 0.5 * dt * (current_yz[1:-2, 0]) / epsilon_0
    Gplus[1:] = Gplus[:-1] - 0.5 * dt * (current_yz[2:-1, 1]) / epsilon_0
    Gminus[:-1] = Gminus[1:] - 0.5 * dt * (current_yz[1:-2, 1]) / epsilon_0

    new_electric_field = np.zeros_like(electric_field)
    new_magnetic_field = np.zeros_like(magnetic_field)

    new_electric_field[:, 0] = Fplus + Fminus
    new_electric_field[:, 1] = Gplus + Gminus
    new_magnetic_field[:, 0] = (Gplus - Gminus) / c
    new_magnetic_field[:, 1] = (Fplus - Fminus) / c

    return new_electric_field, new_magnetic_field


def BunemanLongitudinalSolver(electric_field, current_x, dt, epsilon_0):
    return electric_field - dt / epsilon_0 * current_x[:-1]
