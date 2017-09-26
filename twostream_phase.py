# coding=utf-8
from pythonpic import plotting_parser
from pythonpic.helper_functions.physics import did_it_thermalize
from pythonpic.configs.run_twostream import initial
from pythonpic.visualization import animation, static_plots


args = plotting_parser("Two stream instability")

S = initial("TS_UNSTABLE_LARGE",
            v0 = 0.01,
            N_electrons=25000,
            plasma_frequency=0.1,
            T = 6000,
            ).lazy_run().phase_1d(*args)
