# # coding=utf-8
# import matplotlib.pyplot as plt
# import numpy as np
# import pytest
#
# from ..algorithms.field_interpolation import PeriodicInterpolateField
# from ..classes import Species, PeriodicGrid
#
# @pytest.mark.parametrize("power", range(6))
# def test_poly(power, plotting=False):
#     NG = 16
#     NG_plot = 500
#     L = 1
#
#     x, dx = np.linspace(0, L, NG, retstep=True, endpoint=False)
#
#     N = 128
#     x_particles = np.linspace(0, L, N, endpoint=False)
#
#     def electric_field_function(x):
#         return x ** power
#
#     electric_field = electric_field_function(x)
#
#     interpolated = PeriodicInterpolateField(x_particles, electric_field, x, dx)
#     analytical = electric_field_function(x_particles)
#
#     region_before_last_point = x_particles < x.max()
#
#     def plot():
#         x_plot = np.linspace(0, L, NG_plot, endpoint=False)
#         electric_field_plot = electric_field_function(x_plot)
#         plt.plot(x_plot, electric_field_plot, lw=5)
#         plt.plot(x[region_before_last_point], electric_field[region_before_last_point])
#         plt.plot(x_particles, interpolated, "go-")
#         plt.vlines(x, electric_field.min(), electric_field.max())
#         plt.show()
#         return "poly test failed for power = {}".format(power)
#
#     if plotting:
#         plot()
#
#     assert np.allclose(analytical[region_before_last_point], interpolated[region_before_last_point], atol=1e-2, rtol=1e-2), plot()
#
#
# @pytest.mark.parametrize("field", [lambda x: np.sin(2 * np.pi * x), lambda x: np.cos(2 * np.pi * x)])
# def test_periodic(field, plotting=False):
#     NG = 16
#     NG_plot = 500
#     L = 1
#
#     x, dx = np.linspace(0, L, NG, retstep=True, endpoint=False)
#
#     N = 128
#     x_particles = np.linspace(0, L, N, endpoint=False)
#
#     electric_field = field(x)
#     interpolated = PeriodicInterpolateField(x_particles, electric_field, x, dx)
#     analytical = field(x_particles)
#
#     def plot():
#         x_plot = np.linspace(0, L, NG_plot, endpoint=False)
#         electric_field_plot = field(x_plot)
#         plt.plot(x_plot, electric_field_plot, lw=5)
#         plt.plot(x, electric_field)
#         plt.plot(x_particles, interpolated, "go-")
#         plt.vlines(x, electric_field.min(), electric_field.max())
#         plt.show()
#         return "periodic test failure"
#
#     if plotting:
#         plot()
#
#     assert np.allclose(interpolated, analytical, atol=1e-2, rtol=1e-2), plot()
#
#
# @pytest.mark.parametrize("power", range(2, 6))
# def test_single_particle(power, plotting=False):
#     """tests interpolation of field to particles:
#         at cell boundary
#         at hall cell
#         at 3/4 cell
#         at end of simulation region (PBC)
#     """
#     NG = 16
#     L = 1
#     g = PeriodicGrid(1, L=L, NG=NG)
#     s = Species(1, 1, 4, g)
#
#     def electric_field_function(x):
#         return x ** power
#
#     electric_field = electric_field_function(g.x)
#
#     interpolated = PeriodicInterpolateField(s.x, electric_field, g.x, g.dx)
#     analytical = electric_field_function(s.x)
#     # analytical[-1] = (electric_field[0] + electric_field[-1]) / 2
#
#     def plot():
#         plt.plot(s.x, interpolated, "go-")
#         plt.vlines(g.x, electric_field.min(), electric_field.max())
#         plt.show()
#         return "poly test failed for power = {}".format(power)
#
#     if plotting:
#         plot()
#
#     assert np.allclose(analytical, interpolated), plot()
#
#
# if __name__ == "__main__":
#     test_single_particle()
