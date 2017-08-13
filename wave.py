# coding=utf-8
from pythonpic import plotting_parser, BoundaryCondition
from pythonpic.configs.run_wave import initial, plots


args = plotting_parser("Wave propagation")
for filename, boundary_function in zip(["Wave", "Envelope", "Laser"],
                                       [BoundaryCondition.Laser(1, 1e-6, 1e-5/2, 2e-6).laser_wave,
                                        BoundaryCondition.Laser(1, 1e-6, 1e-5/2, 2e-6).laser_envelope,
                                        BoundaryCondition.Laser(1, 1e-6, 1e-5/2, 2e-6).laser_pulse,
                                        ]):
    s = initial(filename, bc=boundary_function).lazy_run()
    plots(s, *args, alpha=0.5)