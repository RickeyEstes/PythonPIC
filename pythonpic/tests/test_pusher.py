# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pythonpic.configs.run_laser import initial, npic
from pythonpic.helper_functions.physics import electric_charge, electron_rest_mass

from ..helper_functions import physics
from ..algorithms import particle_push
from ..classes import Species, Particle, Simulation
from pythonpic.classes import PeriodicTestGrid, NonperiodicTestGrid
from pythonpic.classes import TestSpecies as Species

from pythonpic.visualization.plotting import plots
from pythonpic.visualization import animation
atol = 1e-1
rtol = 1e-4


@pytest.fixture(params=[1, 2, 3, 10, 100])
def _N_particles(request):
    return request.param


@pytest.fixture(params=[particle_push.boris_push])
def _pusher(request):
    return request.param


@pytest.fixture(params=np.linspace(0.1, 0.9, 10))
def _v0(request):
    return request.param


@pytest.fixture()
def g():
    T = 10
    dt = T / 200
    c = 1
    dx = c * dt
    NG = 10
    L = NG * dx

    g = PeriodicTestGrid(T, L, NG)
    return g

@pytest.fixture()
def g_aperiodic():
    g = NonperiodicTestGrid(T=3, L=1, NG=10)
    return g

def plot(t, analytical_result, simulation_result,
         message="Difference between analytical and simulated results!"):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(t, analytical_result, "b-", label="analytical result")
    ax1.plot(t, simulation_result, "ro--", label="simulation result")

    ax2.plot(t, np.abs(simulation_result - analytical_result), label="difference")
    ax2.plot(t, np.ones_like(t) * (atol + rtol * np.abs(analytical_result)))

    ax2.set_xlabel("t")
    ax1.set_ylabel("result")
    ax2.set_ylabel("difference")
    for ax in [ax1, ax2]:
        ax.grid()
        ax.legend()
        ax.set_xticks(t)
    plt.show()
    return message


def test_constant_field(g, _pusher, _N_particles):
    s = Species(1, 1, _N_particles, g, pusher=_pusher,
                individual_diagnostics=True)
    t = np.arange(0, g.T, g.dt * s.save_every_n_iterations) - g.dt / 2

    def uniform_field(x):
        return np.array([[1, 0, 0]], dtype=float), np.array([[0, 0, 0]], dtype=float)

    x_analytical = 0.5 * (t + g.dt / 2) ** 2 + 0
    s.velocity_push(uniform_field, -0.5)
    for i in range(g.NT):
        s.save_particle_values(i)
        s.velocity_push(uniform_field)
        s.position_push()

    s.save_particle_values(g.NT-1)
    assert np.allclose(s.position_history[:, 0], x_analytical, atol=atol, rtol=rtol), \
        plot(t, x_analytical, s.position_history[:, 0])
    return Simulation(g, [s])


# noinspection PyUnresolvedReferences
def test_relativistic_constant_field(g, _N_particles):
    s = Species(1, 1, _N_particles, g, individual_diagnostics=True)
    t = np.arange(0, g.T, g.dt * s.save_every_n_iterations) - g.dt / 2

    def uniform_field(x):
        return np.array([[1, 0, 0]], dtype=float), np.array([[0, 0, 0]], dtype=float)

    v_analytical = (t - g.dt / 2) / np.sqrt((t - g.dt / 2) ** 2 + 1)
    s.velocity_push(uniform_field, -0.5)
    for i in range(g.NT):
        s.save_particle_values(i)
        s.velocity_push(uniform_field)
        s.position_push()

    s.save_particle_values(g.NT-1)
    assert (s.velocity_history < 1).all(), plot(t, v_analytical, s.velocity_history[:, 0, 0],
                                                f"Velocity went over c! Max velocity: {s.velocity_history.max()}")
    assert np.allclose(s.velocity_history[:, 0, 0], v_analytical, atol=atol, rtol=rtol), \
        plot(t, v_analytical, s.velocity_history[:, 0, 0], )
    return Simulation(g, [s])


# noinspection PyUnresolvedReferences
def test_relativistic_magnetic_field(g, _N_particles, _v0):
    B0 = 1
    s = Species(1, 1, _N_particles, g, individual_diagnostics=True)
    t = np.arange(0, g.T, g.dt * s.save_every_n_iterations) - g.dt / 2
    s.v[:, 1] = _v0

    def uniform_magnetic_field(x):
        return np.array([[0, 0, 0]], dtype=float), np.array([[0, 0, B0]], dtype=float)

    gamma = physics.gamma_from_v(s.v, s.c)[0,0]
    vy_analytical = _v0 * np.cos(s.q * B0 * (t - g.dt / 2) / (s.m * gamma))

    s.velocity_push(uniform_magnetic_field, 0.5)
    for i in range(g.NT):
        s.save_particle_values(i)
        s.velocity_push(uniform_magnetic_field)
        s.position_push()
    assert (s.velocity_history < g.c).all(), plot(t, vy_analytical, s.velocity_history[:, 0, 1],
                                                f"Velocity went over c! Max velocity: {s.velocity_history.max()}")
    assert np.allclose(s.kinetic_energy_history[1:-1], s.kinetic_energy_history[1:-1].mean(), atol=atol, rtol=rtol), "Energy is off!"

    assert np.allclose(s.velocity_history[:, 0, 1], vy_analytical, atol=atol, rtol=rtol)
    return Simulation(g, [s])


# noinspection PyUnresolvedReferences
@pytest.mark.parametrize("E0", np.linspace(-10, 10, 10))
def test_relativistic_harmonic_oscillator(g, _N_particles, E0):
    E0 = 1
    omega = 2 * np.pi / g.T
    s = Species(1, 1, _N_particles, g, individual_diagnostics=True)
    t = np.arange(0, g.T, g.dt * s.save_every_n_iterations) - g.dt / 2

    t_s = t - g.dt / 2
    v_analytical = E0 * s.c * s.q * np.sin(omega * t_s) / np.sqrt(
        (E0 * s.q * np.sin(omega * t_s)) ** 2 + (s.m * omega * s.c) ** 2)

    def electric_field(x, t):
        return np.array([[1, 0, 0]], dtype=float) * E0 * np.cos(omega * t), np.array([[0, 0, 0]], dtype=float)

    s.velocity_push(lambda x: electric_field(x, 0), 0.5)
    for i in range(g.NT):
        s.save_particle_values(i)
        s.velocity_push(lambda x: electric_field(x, i * g.dt))
        s.position_push()

    s.save_particle_values(g.NT-1)
    assert (s.velocity_history < 1).all(), plot(t, v_analytical, s.velocity_history[:, 0, 0],
                                                f"Velocity went over c! Max velocity: {s.velocity_history.max()}")
    assert np.allclose(s.velocity_history[:, 0, 0], v_analytical, atol=atol, rtol=rtol), \
        plot(t, v_analytical, s.velocity_history[:, 0, 0], )

@pytest.mark.parametrize(["v0", "gamma"], [
    [3/5, 1.25],
    [4/5, 5/3],
     ])
def test_gamma(v0, gamma):
    v = np.array([[v0, 0, 0]])
    assert np.isclose(physics.gamma_from_v(v, 1), gamma)

def no_field(x):
    return np.array([[0, 0, 0]], dtype=float), np.array([[0, 0, 0]], dtype=float)

@pytest.mark.parametrize(["v0", "expected_kin"], [
    [3/5, 0.25],
    [4/5, 0.666666],
    ])
def test_kinetic_energy(g, v0, expected_kin, _N_particles):
    s = Particle(g, 0, v0, scaling=_N_particles)
    s.velocity_push(no_field)
    total_expected_kin = expected_kin * _N_particles * g.c**2
    assert np.isclose(total_expected_kin, s.energy)
    assert np.isclose(total_expected_kin, s.kinetic_energy)



@pytest.mark.parametrize("v0", [
    0.9,
    0.99,
    0.999,
    0.9999,
    ])
def test_high_relativistic_velocity(g, v0):
    s = Particle(g, 0, v0)
    t = np.arange(0, g.T, g.dt * s.save_every_n_iterations) - g.dt / 2


    s.velocity_push(lambda x: no_field(x), -0.5)
    for i in range(g.NT):
        s.save_particle_values(i)
        s.velocity_push(lambda x: no_field(x))
        s.position_push()

    s.save_particle_values(g.NT-1)
    expected_kinetic_energy = (physics.gamma_from_v(s.v, g.c) - 1).sum() * s.eff_m * g.c**2
    assert np.isclose(s.kinetic_energy_history[1: -1].mean(), expected_kinetic_energy)
    assert np.allclose(s.kinetic_energy_history[1:-1], s.kinetic_energy_history[1:-1].mean(), atol=atol, rtol=rtol), "Energy is off!"
    assert (s.velocity_history < 1).all(), f"Velocity went over c! Max velocity: {s.velocity_history.max()}"

@pytest.mark.parametrize("v0", [0.9, 0.99, 0.999, 0.9999])
def test_high_relativistic_velocity_multidirection(g, v0):
    s = Particle(g, 0, v0*0.5, v0*0.5, v0*0.5)
    t = np.arange(0, g.T, g.dt * s.save_every_n_iterations) - g.dt / 2

    def no_field(x):
        return np.array([[0, 0, 0]], dtype=float), np.array([[0, 0, 0]], dtype=float)

    s.velocity_push(lambda x: no_field(x), -0.5)
    for i in range(g.NT):
        s.save_particle_values(i)
        s.velocity_push(lambda x: no_field(x))
        s.position_push()

    s.save_particle_values(g.NT-1)
    assert np.allclose(s.kinetic_energy_history[1:-1], s.kinetic_energy_history[1:-1].mean(), atol=atol, rtol=rtol), "Energy is off!"
    assert (s.velocity_history < 1).all(), f"Velocity went over c! Max velocity: {s.velocity_history.max()}"

def test_periodic_particles(g):
    s = Species(1, 1, 100, g, individual_diagnostics=True)
    s.distribute_uniformly(g.L)
    s.v[:] = 0.5
    for i in range(g.NT):
        force = lambda x: (np.array([[0, 0, 0]], dtype=float), np.array([[0, 0, 0]], dtype=float))
        s.velocity_push(force)
        s.position_push()
        s.apply_bc()
    assert s.N_alive == s.N, "They're dead, Jim."

def test_nonperiodic_particles(g_aperiodic):
    g = g_aperiodic
    s = Species(1, 1, 100, g, individual_diagnostics=True)
    s.distribute_uniformly(g.L)
    s.v[:] = 0.5
    for i in range(g.NT):
        force = lambda x: (np.array([[0, 0, 0]], dtype=float), np.array([[0, 0, 0]], dtype=float))
        s.velocity_push(force)
        s.position_push()
        s.apply_bc()
    assert s.N_alive == 0


# @pytest.mark.parametrize("T", [1.05],)
# def test_energy_conservation_electron(T):
#     g = NonperiodicTestGrid(T=T, L=1, NG=100, c=1)
#     electron = Particle(g, 2*g.dx, g.c*0.99, q = -1, m =1e-3, name="electron")
#     filename = f"test_energy_conservation_electron_{T}"
#     sim = Simulation(g, [electron], category_type="test", filename=filename)
#     sim.run(init=False).postprocess()
#     # plt.plot(electron_momentum[:,:,0])
#     # plt.plot(proton_momentum[:,:,0])
#     # plt.plot((electron_momentum + proton_momentum)[:,:,0])
#     # plt.plot(electron.momentum_history[:,:,0])
#     # plt.show()
#     assert False, plots(sim, show_animation=True, show_static=True, animation_type=animation.OneDimAnimation, frames="all")
#
# @pytest.mark.parametrize("T", [10],)
# def test_energy_conservation(T):
#      g = NonperiodicTestGrid(T=T, L=10, NG=1000, c=1)
#      electron = Particle(g, g.L/2, g.c*0.99, q = -1, m =1e-3, name="electron")
#      proton = Particle(g, g.L/2, 0, q = 1, m =2, name="proton")
#      filename = f"test_energy_conservation_{T}"
#      sim = Simulation(g, [electron, proton], category_type="test", filename=filename)
#      sim.run(init=False).postprocess()
#      plt.plot(electron.momentum_history[:,:,0])
#      plt.plot(proton.momentum_history[:,:,0])
#      plt.plot((electron.momentum_history + proton.momentum_history)[:,:,0])
#      plt.show()
#      assert False, plots(sim, show_animation=True, show_static=True, animation_type=Animation.OneDimAnimation, frames="all")

def test_laser_pusher():
    S = initial("test_current", 0, 1378, 0, 0, 0)
    p = Particle(S.grid,
                 9.45*S.grid.dx,
                 0,
                 0,
                 q=-electric_charge,
                 m=electron_rest_mass,
                 scaling=npic)
    S.grid.list_species = [p]
    Ev, Bv = (np.array([[0, 1.228e12, 0]]), np.array([[0,0,4.027e3]]))
    E = lambda x: (Ev, Bv)
    p.velocity_push(E)
    p.position_push()
    print(p.v)
    expected_u = np.array([[5.089e4, -5.5698e6, 0]]) # this is u!
    expected_v = expected_u #* gamma_from_u(expected_u, S.grid.c)
    print(expected_v)
    print((p.v - expected_v)[0,:2] / p.v[0,:2] * 100, "%")
    # print(f"vx:{a:.9e}\n"
    #       f"vy:{b:.9e}\n"
    #       f"vz:{c:.9e}\n"
    #       f"KE:{e:.9e}\n")
    assert np.allclose(expected_v, p.v, atol=1e-1, rtol=1e-2)
