# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np

from ..algorithms.charge_deposition import density_deposition, periodic_density_deposition


def test_single_particle(plotting=False):
    NG = 8
    L = 1
    x, dx = np.linspace(0, L, NG, retstep=True, endpoint=False)
    # REFACTOR: gridify

    q = 1

    x_particles = np.array([x[3] + dx / 2, x[5] + 0.75 * dx])
    analytical_charge_density = x_particles.size * q / L

    indices = (x_particles // dx).astype(int)
    print(indices)

    analytical_charge_density = np.array([0., 0., 0., 0.5, 0.5, 0.25, 0.75, 0., 0.])
    charge_density = q * density_deposition(x, dx, x_particles)
    print(f"charge density {charge_density}")
    print(f"analytical charge density {analytical_charge_density}")

    def plot():
        plt.plot(x, charge_density, "bo-", label="scattered")
        plt.plot(x, analytical_charge_density, "go-", label="analytical")
        plt.plot(x_particles, q * np.ones_like(x_particles) / x_particles.size, "r*", label="particles")
        plt.legend()
        plt.show()
        return "single particle interpolation is off!"

    if plotting:
        plot()
    assert np.allclose(charge_density, analytical_charge_density), plot()


def test_constant_density(plotting=False):
    NG = 8
    L = 1
    # REFACTOR: gridify
    x, dx = np.linspace(0, L, NG, retstep=True, endpoint=False)

    q = 1

    N = 128
    x_particles = np.linspace(0, L, N, endpoint=False)
    analytical_charge_density = x_particles.size * q / L / NG * np.ones_like(x)

    charge_density = periodic_density_deposition(x, dx, x_particles)
    print(f"charge density {charge_density}")
    print(f"analytical charge density {analytical_charge_density}")

    def plot():
        plt.plot(x, charge_density[:-1], "bo-", label="scattered")
        plt.plot(x, analytical_charge_density, "go-", label="analytical uniform")
        plt.plot(x_particles, q * 2 / dx * np.ones_like(x_particles), "r*", label="particles")
        plt.legend()
        plt.show()
        return False

    if plotting:
        plot()
    assert np.allclose(charge_density[:-1], analytical_charge_density), plot()


def test_boundaries(plotting=False):
    NG = 8
    L = 1
    x, dx = np.linspace(0, L, NG, retstep=True, endpoint=False)
    # REFACTOR: gridify

    q = 1

    x_particles = np.array([x[3] + dx / 2, x[-1] + 0.25 * dx])
    analytical_charge_density = x_particles.size * q / L

    analytical_charge_density = np.array([0.25, 0., 0., 0.5, 0.5, 0., 0., 0.75])
    charge_density = periodic_density_deposition(x, dx, x_particles)
    print(f"charge density {charge_density}")
    print(f"analytical charge density {analytical_charge_density}")

    def plot():
        plt.plot(x, charge_density[:-1], "bo-", label="scattered")
        plt.plot(x, analytical_charge_density, "go-", label="analytical")
        plt.plot(x_particles, q * np.ones_like(x_particles) / x_particles.size, "r*", label="particles")
        plt.legend(loc='best')
        plt.show()
        return "single particle interpolation is off!"

    if plotting:
        plot()
    assert np.allclose(charge_density[:-1], analytical_charge_density), plot()

# def test_uniform_current_deposition(plots=False):
#     """
#      N particles
#      charge q
#      velocity v
#     """
#
#     g = RelativisticGrid()
#     p = Species(1, 1, 128, "p")
#     p.v[:, 0] = 0
#     p.v[:, 1] = -1
#     p.v[:, 2] = 1
#     p.distribute_uniformly(g.L, g.dx / 1000 * np.pi)
#     g.gather_current([p])
#     if plots:
#         plt.plot(g.x, g.current_density)
#         plt.show()
#
#     predicted_values = p.q * p.v.sum(axis=0) / g.L * g.dx
#     for dim, val in zip(range(3), predicted_values):
#         assert np.isclose(g.current_density[:, dim], val).all(), (g.current_density[0, :], predicted_values)


# def test_nonuniform_current_deposition(plots=False):
#     """
#      N particles
#      charge q
#      velocity v
#     """
#
#     g = RelativisticGrid()
#     p = Species(1, 1, 128, "p")
#     dims = np.arange(3)
#     p.v[:, dims] = np.arange(p.N)[:, np.newaxis] ** dims[np.newaxis, :]
#     p.distribute_uniformly(g.L, g.dx / 1000 * np.pi)
#     g.gather_current([p])
#     if plots:
#         plt.plot(g.x, g.current_density)
#         plt.show()
#
#     predicted_values = p.q * p.v.sum(axis=0) / g.L * g.dx
#     for dim, val in zip(dims, predicted_values):
#         indices = np.arange(1, g.NG)
#         fit = np.polyfit(g.x[indices], g.current_density[indices, dim], 2)
#         fit /= np.linalg.norm(fit)
#         print(fit)
#         if plots:
#             plt.plot(g.x, np.polyval(fit, g.x))
#             plt.plot(g.x, g.current_density[:, dim])
#             plt.show()
#         assert np.isclose(fit[2 - dim], 1, rtol=1e-4), (fit[dim])


# def test_current_backup():
#     from Species import Species
#     import matplotlib.pyplot as plt
#     NT=100
#     NG = 32
#     NP = 20000
#     print(NP/NG/NG)
#     g = PeriodicGrid(NT=NT, NG=NG,relativistic=True)
#     normalization = NP * 1 / NG
#     g.current_density = np.ones_like(g.current_density) * 0.5 * normalization
#     g.current_density[:,1] = 1 * normalization
#     g.current_density[:,2] = -1 * normalization
#     print(g.current_density.shape)
#
#     g2 = PeriodicGrid(NG = NG, NT=NT, relativistic=True)
#     s = Species(1,1,NP, "particles", NT)
#     s.distribute_uniformly(g2.L,g2.dx/2.5/g2.NG)
#     s.v = np.ones_like(s.v)*0.5
#     s.v[:,1] = 1
#     s.v[:,2] = -1
#     g2.gather_current([s])
#
#     def plot(g, gridname):
#         fig = plt.figure()
#         labels = [gridname + direction for direction in ["jx", "jy", "jz"]]
#         lines = plt.plot(g.x, g.current_density, "o-")
#         plt.legend(lines, labels)
#         plt.grid()
#
#     plot(g, "g set")
#     plot(g2, "g deposited")
#
#
#     def plot_diff(g, g2, s):
#         fig = plt.figure()
#         diff = (g.current_density-g2.current_density)
#         labels = ["difference in " + direction for direction in ["jx", "jy", "jz"]]
#         lines = plt.plot(g.x, diff, "o-")
#         plt.plot(s.x, np.zeros_like(s.x), "o")
#         plt.legend(lines, labels)
#         plt.grid()
#         plt.vlines(g.x, diff.min()*1.5, diff.max()*1.5)
#         return diff.max()
#     print(plot_diff(g,g2,s))
#     plt.ylim(-0.04, 0.04)
#
#     plt.show()
#     # plt.show()
