# coding=utf-8
import numpy as np
import scipy.integrate

from pythonpic.algorithms import helper_functions
profiles = {"linear": lambda x: x,
            "quadratic": lambda x: x ** 2,
            "exponential": lambda x: np.exp(10 * (x - 1))}


def FDENS(x, moat_left, ramp_length, plasma_length, N, func='linear'):
    func = profiles[func]
    rectangle_area = (plasma_length - ramp_length)
    modified_func = lambda x_value: func((x_value - moat_left) / ramp_length)
    ramp_area, _ = scipy.integrate.quad(modified_func, moat_left, moat_left + ramp_length)
    # triangle_area = 0.5 * ramp_length
    normalization = N / (rectangle_area + ramp_area)
    result = np.zeros_like(x)
    region1 = x < moat_left
    region2 = (x < moat_left + ramp_length) & ~region1
    region3 = (x < moat_left + plasma_length) & ~(region2 | region1)
    result[region2] = normalization * modified_func(x[region2])
    result[region3] = normalization
    return result

def relativistic_maxwellian(v, N, c, m, T):
    p = 1
    gamma = helper_functions.gamma_from_v(v, c)
    kinetic_energy = (gamma - 1) * m * c ** 2
    normalization = N / (2 * np.pi) * m * c **2 / T / (1 + T / m / c**2)
    f = normalization * np.exp(-kinetic_energy/T)
    # TODO: WORK IN PROGRESS


def generate(dense_range, func, *function_params):
    y = func(dense_range, *function_params)
    integrated = scipy.integrate.cumtrapz(y, dense_range, initial=0).astype(int)
    indices = np.diff(integrated) == 1
    print(f"sum of indices: {indices.sum()}")
    return dense_range[:-1][indices]