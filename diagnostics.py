import numpy as np
from constants import epsilon_0


def energies(r, v, m, dx, rho, phi):
    particle_kinetic_energy = 0.5 * m * np.sum(v * v)
    field_potential_energy = 0.5 * epsilon_0 * dx * np.sum(rho * phi)
    total_energy = particle_kinetic_energy + field_potential_energy
    return particle_kinetic_energy, field_potential_energy, total_energy


def L2norm(A, B):
    return np.sum((A - B)**2) / np.sum(A**2)
