# coding=utf-8
import numpy as np
import pytest
import matplotlib.pyplot as plt
from pythonpic.classes import Species
from pythonpic.classes import TestSpecies as Species
from pythonpic.classes import NonperiodicTestGrid


@pytest.mark.parametrize('func', [lambda x: x + 3, lambda x: x**2, lambda x: np.sin(2*np.pi*x)])
def test_interpolation(func):
    def func2(x):
        return func(x) + 100
    g = NonperiodicTestGrid(1, 1, 100)
    wide_g = (np.arange(g.NG+2)-1)*g.dx
    g.electric_field[:,0] = func2(wide_g)
    s = Species(1, 1, 1000, g)
    s.distribute_uniformly(g.L)
    interpolated_field = g.field_function(s.x)[0][:,0]
    expected_field = func2(s.x)
    lnorm = np.abs((interpolated_field - expected_field) / expected_field).sum()
    print(lnorm)
    def plot():
        plt.plot(wide_g, g.electric_field[:, 0], "+", label="grid")
        plt.plot(s.x, expected_field, ".", label="analytical particles")
        plt.plot(s.x, interpolated_field, label="interpolated" "o")
        plt.legend()
        plt.show()
    assert np.allclose(expected_field, interpolated_field), plot()

