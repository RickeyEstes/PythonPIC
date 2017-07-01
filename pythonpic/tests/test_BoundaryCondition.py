# coding=utf-8

import pytest
import numpy as np
from matplotlib import pyplot as plt
from pythonpic.algorithms import BoundaryCondition

laser = BoundaryCondition.LaserCircular(1, 1)
t = np.linspace(0, 10, 1000)

def test_polarization_phase():
    phase = laser.polarisation_phase(t)
    assert np.allclose(t*2*np.pi, phase)

def test_field():
    Ex, Ey, Ez = laser.E_values(t)
    assert np.any(Ey > 0)
    assert np.any(Ey < 0)
    assert np.any(Ez > 0)
    assert np.any(Ez < 0)
