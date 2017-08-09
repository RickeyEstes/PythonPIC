# coding=utf-8
from pythonpic import plotting_parser
from pythonpic.configs.run_coldplasma import plots, cold_plasma_oscillations
from pythonpic.visualization import animation
from pythonpic.helper_functions.physics import epsilon_zero, lightspeed


args = plotting_parser("Cold plasma oscillations")
plasma_frequency = 1
push_mode = 2
N_electrons = 1024
NG = 64
qmratio = -1
T = 10
scaling = 1
c = 10
epsilon_zero = 1

S = cold_plasma_oscillations(f"COSCALING", qmratio=qmratio, plasma_frequency=plasma_frequency, NG=NG,
                             N_electrons=N_electrons, epsilon_zero=epsilon_zero, push_mode=push_mode,
                             save_data=False, T = T, scaling=scaling, c=c).run()
plots(S, *args, animation_type = animation.OneDimAnimation, alpha=0.3)
