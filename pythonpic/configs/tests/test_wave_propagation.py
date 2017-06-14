"""Tests the Leapfrog wave PDE solver"""
# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import pytest

from . import on_failure
from pythonpic.algorithms import BoundaryCondition
from ..run_wave import wave_propagation
from pythonpic.visualization.plotting import plots


def plot_all(field_history, analytical_solution):
    T, X = field_history.shape
    XGRID, TGRID = np.meshgrid(np.arange(X), np.arange(T))
    for n in range(field_history.shape[0]):
        if field_history.shape[0] < 200 or n % 100:
            plt.plot(field_history[n])
    fig = plt.figure()
    ax = fig.add_subplot(211)
    CF1 = ax.contourf(TGRID, XGRID, field_history, alpha=1)
    ax.set_xlabel("time")
    ax.set_ylabel("space")
    plt.colorbar(CF1)
    ax2 = fig.add_subplot(212)
    CF2 = ax2.contourf(TGRID, XGRID, analytical_solution, alpha=1)
    ax2.set_xlabel("time")
    ax2.set_ylabel("space")
    plt.colorbar(CF2)
    plt.show()

@pytest.fixture(scope="module", params=[lambda x: x.laser_wave, lambda x: x.laser_envelope, lambda x: x.laser_pulse])
def shape(request):
    return request.param

@pytest.fixture(scope="module", params=[1, 1000, 1e23])
def intensity(request):
    return request.param

@pytest.fixture(scope="module", params=[1, 3, 0.5])
def wavelength(request):
    return request.param

@pytest.fixture(scope="module", params=range(1, 6, 2))
def power(request):
    return request.param

@pytest.fixture(scope="module")
def wave_propagation_helper(shape, intensity, wavelength, power):
    laser = BoundaryCondition.Laser(intensity, wavelength, 10, power)
    bc = shape(laser)
    filename = f"wave_propagation_test_I{intensity}L{wavelength}P{power}"
    sim = wave_propagation(filename, bc).run().postprocess()
    return sim, laser


def test_amplitude(wave_propagation_helper):
    run, laser = wave_propagation_helper
    amplitude = laser.laser_amplitude
    max_efield = np.abs(run.grid.electric_field_history).max()
    assert max_efield < amplitude, plots(run, *on_failure)
    # assert max_efield > amplitude*2**-0.5, plots(run, *on_failure)

def test_wave_propagation(wave_propagation_helper):
    run, laser = wave_propagation_helper
    mean_energy = run.grid.grid_energy_history[...].mean()
    assert mean_energy, plots(run, *on_failure)


def test_polarization_orthogonality(wave_propagation_helper):
    run, laser = wave_propagation_helper
    angles = ((run.grid.electric_field_history[...] * run.grid.magnetic_field_history[...]).sum(axis=(1, 2)))
    assert np.isclose(angles, 0).all(), "Polarization is not orthogonal!"



