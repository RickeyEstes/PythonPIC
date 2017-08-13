# coding=utf-8
from pythonpic import plotting_parser
from pythonpic.configs.run_coldplasma import plots, initial
from pythonpic.visualization import animation
from pythonpic.helper_functions.physics import epsilon_zero, lightspeed


args = plotting_parser("Cold plasma oscillations")
plasma_frequency = 1
push_mode = 2
N_electrons = 1024 * 4
NG = 256
qmratio = -1
T = 50
scaling = 1
c = 10
epsilon_zero = 1

S = initial(f"CO_LINEAR", qmratio=qmratio, plasma_frequency=plasma_frequency, NG=NG,
            N_electrons=N_electrons, epsilon_zero=epsilon_zero, push_mode=push_mode,
            push_amplitude=0.1,
            save_data=False, T = T, scaling=scaling, c=c).run()
plots(S, *args, animation_type = animation.OneDimAnimation, alpha=0.3)

S = initial(f"CO_NONLINEAR", qmratio=qmratio, plasma_frequency=plasma_frequency, NG=NG,
            N_electrons=N_electrons, epsilon_zero=epsilon_zero, push_mode=push_mode,
            push_amplitude=0.8,
            save_data=False, T = T, scaling=scaling, c=c).run()
plots(S, *args, animation_type = animation.OneDimAnimation, alpha=0.3)

