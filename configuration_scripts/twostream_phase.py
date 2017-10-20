# coding=utf-8
from pythonpic import plotting_parser
from pythonpic.configs.run_twostream import initial

args = plotting_parser("Two stream instability")

S = initial("TS_UNSTABLE_LARGE",
            v0 = 0.01,
            N_electrons=25000,
            plasma_frequency=0.1,
            T = 6000,
            ).lazy_run().phase_1d(*args)
