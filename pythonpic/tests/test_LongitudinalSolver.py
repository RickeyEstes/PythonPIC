# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ..classes import Simulation
from pythonpic.classes import PeriodicTestGrid
from ..visualization.time_snapshots import FieldPlot, CurrentPlot
from pythonpic.classes import TestSpecies as Species

@pytest.fixture(params=[10, 100], scope='module')
def NG(request):
    return request.param

@pytest.fixture(params=[1, 10], scope='module')
def T(request):
    return request.param

@pytest.fixture(params=[1, 2], scope='module')
def c(request):
    return request.param

@pytest.fixture(params=[0.1, 1], scope='module')
def L(request):
    return request.param

@pytest.fixture(params=[1, 100], scope='module')
def epsilon_0(request):
    return request.param

def test_empty_grid(T, L, NG, c, epsilon_0):
    g = PeriodicTestGrid(T, L, NG, c, epsilon_0)
    g.electric_field[:,0] = 1
    s = Simulation(g)
    s.run(init=False)
    assert np.allclose(g.electric_field_history[:,1:,0], 1), "Current-less grid gathers current somehow."

def test_current_at_grid(T, L, NG, c, epsilon_0):
    g = PeriodicTestGrid(T, L, NG, c, epsilon_0)
    index = int(NG/3)
    g.current_density_x[index+1] = 1
    for i in range(g.NT):
        g.save_field_values(i)
        g.solve()
    expected_field = -1 * g.dt / g.epsilon_0 * np.arange(g.NT)
    achieved_field = g.electric_field_history[:,index,0]
    assert np.allclose(achieved_field, expected_field), "Long. field does not grow linearly"
