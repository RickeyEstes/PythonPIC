# Algorithms for deposition of charge and currents on grid
# coding=utf-8
import numpy as np


def density_deposition(x, dx: float, x_particles):
    """scatters charge from particles to grid
    uses linear interpolation
    x_i | __________p___| x_i+1
    for a particle $p$ in cell $i$ of width $dx$ the location in cell is defined as
    $$X_p = x_p - x_i$$
    then, $F_r = X_p/dx$ is the fraction of charge going to the right side of the cell
    (as $X_p \to dx$, the particle is closer to the right cell)
    while $F_l = 1-F_r$ is the fraction of charge going to the left

    numpy.bincount is used to count particles in cells
    the weights are the fractions for each cell

    to change the index for the right going  (keeping periodic boundary conditions)
    numpy.roll is used
    """
    logical_coordinates = (x_particles / dx).astype(int)
    charge_to_right = x_particles / dx - logical_coordinates
    charge_to_left = 1 - charge_to_right
    charge_hist_to_right = np.bincount(logical_coordinates+1, charge_to_right, minlength=x.size+1)
    charge_hist_to_left = np.bincount(logical_coordinates, charge_to_left, minlength=x.size+1)
    return charge_hist_to_right + charge_hist_to_left


def periodic_density_deposition(x, dx: float, x_particles):
    result = density_deposition(x, dx, x_particles)
    result[0] += result[-1]
    return result
