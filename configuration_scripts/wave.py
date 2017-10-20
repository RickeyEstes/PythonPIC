# coding=utf-8
from pythonpic import plotting_parser, BoundaryCondition
from pythonpic.configs.run_wave import initial


args = plotting_parser("Wave propagation")
for filename, boundary_function in zip(["Wave", "Envelope", "Laser"],
        [BoundaryCondition.LaserCircular(1, 1e-6, 1e-5/2, 2e-6, bc_function="pulse"),
        BoundaryCondition.LaserCircular(1, 1e-6, 1e-5/2, 2e-6, bc_function="wave"),
        BoundaryCondition.LaserCircular(1, 1e-6, 1e-5/2, 2e-6, bc_function="envelope"),
        ]):
    s = initial(filename, bc=boundary_function).lazy_run().wave_1d(*args)
