# coding=utf-8
import numpy as np
from numba import njit

@njit()
def PeriodicInterpolateField(x_particles, scalar_field, dx: float):
    """gathers field from grid to particles

    the reverse of the algorithm from charge_density_deposition

    there is no need to use numpy.bincount as the map is
    not N (number of particles) to M (grid), but M to N, N >> M
    """
    logical_coordinates = (x_particles / dx).astype(np.int32)
    NG = scalar_field.shape[0]-2
    # the below is reshaped to 2d (N, 1) so as to allow for multiplication
    # with 2d field array
    right_fractions = (x_particles / dx - logical_coordinates).reshape((x_particles.size, 1))
    field = (1 - right_fractions) * scalar_field[logical_coordinates+1] + \
            right_fractions * scalar_field[(logical_coordinates +1) % NG  + 1]
    return field

@njit()
def AperiodicInterpolateField(x_particles, scalar_field, dx: float):
    """gathers field from grid to particles

    the reverse of the algorithm from charge_density_deposition

    there is no need to use numpy.bincount as the map is
    not N (number of particles) to M (grid), but M to N, N >> M
    """
    logical_coordinates = (x_particles / dx).astype(np.int32)
    right_fractions = (x_particles / dx - logical_coordinates).reshape((x_particles.size, 1))
    field = (1 - right_fractions) * scalar_field[logical_coordinates+1] + \
            right_fractions * scalar_field[logical_coordinates + 2]
    return field
