# coding=utf-8
import numpy as np
from scipy.integrate import quad, cumtrapz

from ..helper_functions import physics
profiles = {"linear": lambda x: x,
            "quadratic": lambda x: x ** 2,
            "exponential": lambda x: np.exp(10 * (x - 1))}


def FDENS(x, moat_left, ramp_length, plasma_length, N, func='linear'):
    func = profiles[func]
    rectangle_area = (plasma_length - ramp_length)
    modified_func = lambda x_value: func((x_value - moat_left) / ramp_length)
    # noinspection PyTupleAssignmentBalance
    ramp_area, _ = quad(modified_func, moat_left, moat_left + ramp_length)
    normalization = (N+0.1) / (rectangle_area + ramp_area) # N + 0.1 due to non-exact float calculations
    result = np.zeros_like(x)
    region1 = x < moat_left
    region2 = (x < moat_left + ramp_length) & ~region1
    region3 = (x < moat_left + plasma_length) & ~(region2 | region1)
    result[region2] = normalization * modified_func(x[region2])
    result[region3] = normalization
    return result

# def relativistic_maxwellian(v, N, c, m, T):
#     gamma = physics.gamma_from_v(v, c)
#     kinetic_energy = (gamma - 1) * m * c ** 2
#     normalization = N / (2 * np.pi) * m * c **2 / T / (1 + T / m / c**2)
#     f = normalization * np.exp(-kinetic_energy/T)
#     # TODO: finish this algorithm
#     raise NotImplementedError


def generate(dense_range, func, *function_params):
    y = func(dense_range, *function_params)
    integrated = cumtrapz(y, dense_range, initial=0).astype(int)
    diffs = np.diff(integrated)

    assert (diffs <= 1).all(), "There's two particles in a cell! Increase resolution."
    indices = diffs == 1
    return dense_range[:-1][indices]
