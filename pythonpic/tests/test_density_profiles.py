# coding=utf-8
import numpy as np
import pytest
from matplotlib import pyplot as plt

from ..algorithms import density_profiles
from ..classes import Species, Simulation
from pythonpic.classes import PeriodicTestGrid
from pythonpic.classes import TestSpecies as Species
from ..visualization.time_snapshots import SpatialPerturbationDistributionPlot, SpatialDistributionPlot

@pytest.fixture(params=np.linspace(0.05, 0.3, 3), scope='module')
def _fraction(request):
    return request.param


_second_fraction = _fraction

@pytest.fixture(params=np.linspace(4000, 10000, 2, dtype=int), scope='module')
def _N_particles(request):
    return request.param

@pytest.fixture(params=density_profiles.profiles.keys(), scope='module')
def _profile(request):
    return request.param

# noinspection PyUnresolvedReferences
@pytest.fixture()
def test_density_helper(_fraction, _second_fraction, _profile, _N_particles):

    g = PeriodicTestGrid(1, 100, 100)
    s = Species(1, 1, _N_particles, g)

    moat_length = g.L * _fraction
    ramp_length = g.L * _second_fraction
    plasma_length = 2*ramp_length
    # plasma_length = plasma_length if plasma_length > ramp_length else ramp_length
    s.distribute_nonuniformly(g.L, moat_length, ramp_length, plasma_length, profile=_profile)
    return s, g, moat_length, plasma_length, _profile

@pytest.mark.parametrize("std", [0.00001])
def test_fitness(test_density_helper, std):
    np.random.seed(0)
    s, g, moat_length, plasma_length, profile = test_density_helper
    assert (s.x > moat_length).all(), "Particles running out the right side!"
    assert (s.x <= moat_length + plasma_length).all(), "Particles running out the left side!"
    # particle conservation
    assert s.N == s.x.size, "Particles are not conserved!"
    sim = Simulation(g, [s])
    s.gather_density()
    s.save_particle_values(0)
    s.random_position_perturbation(std)
    assert (s.x > moat_length).all(), "Particles running out the left side!"
    assert (s.x <= moat_length + plasma_length).all(), "Particles running out the right side!"
    s.gather_density()
    s.save_particle_values(1)
    def plots():
        fig, ax = plt.subplots()
        fig.suptitle(f"{s.N} particles with {profile} profile")
        plot = SpatialPerturbationDistributionPlot(sim, ax)
        plot.update(1)
        plot2 = SpatialDistributionPlot(sim, ax)
        ax.plot(g.x, density_profiles.FDENS(g.x, moat_length, plasma_length/2,
                                            plasma_length, s.N, profile))
        plot2.update(0)
        plt.show()
    assert np.allclose(s.density_history[0], s.density_history[1], atol=1e-3), plots()


# @pytest.mark.parametrize("std", [0])
# def test_stability(std):
#     S = laser("stability_test", 1000, 1378, 0, 0, std).test_run()
#     def plots():
#         fig, ax = plt.subplots()
#         fig.suptitle(std)
#         plot = SpatialPerturbationDistributionPlot(S, ax)
#         plot.update(S.NT-1)
#         make_sure_path_exists(S.filename)
#         fig.savefig(S.filename.replace(".hdf5", ".png"))
#         plt.show()
#         plt.close(fig)
#     s = S.list_species[0]
#     assert np.allclose(s.density_history[0], s.density_history[-1]), plots()