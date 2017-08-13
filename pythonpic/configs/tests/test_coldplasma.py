# coding=utf-8

import pytest

from . import on_failure
from pythonpic.helper_functions.physics import get_dominant_mode
from ..run_coldplasma import initial, plots


@pytest.mark.parametrize("push_mode", range(1, 10, 2))
def test_linear_dominant_mode(push_mode):
    """In the linear mode the """
    plasma_frequency = 1
    N_electrons = 1024
    NG = 64
    qmratio = -1

    run_name = f"CO_LINEAR_{push_mode}"
    S = initial(run_name, qmratio=qmratio, plasma_frequency=plasma_frequency, NG=NG,
                N_electrons=N_electrons, push_mode=push_mode).test_run()
    calculated_dominant_mode = get_dominant_mode(S)
    assert (calculated_dominant_mode == push_mode) or (calculated_dominant_mode % push_mode == 0), (
        f"got {calculated_dominant_mode} instead of {push_mode}",
        plots(S, show_animation=True))


# # TODO: this setup is actually unstable
# @pytest.mark.parametrize(["N_electrons", "push_amplitude"],
#                          [(256, 1e-6), (256, 1e-9)])
# def test_kaiser_wilhelm_instability_avoidance(N_electrons, push_amplitude):
#     """aliasing effect with particles exactly at or close to grid nodes.
#     Particles exactly on grid nodes cause excitation of high modes.
#     Even a slight push prevents that."""
#     S = initial(f"CO_KWI_STABLE_{N_electrons}_PUSH_{push_amplitude}",
#                                  N_electrons=N_electrons, NG=256,
#                                  T = 200,
#                                  push_mode=2,
#                                  push_amplitude=push_amplitude).test_run()
#     plots(S, show_animation=True)
#     assert get_dominant_mode(S) == 1, plots(S, show_animation=True)


@pytest.mark.parametrize("N", [128])
def test_kaiser_wilhelm_instability(N):
    # __doc__ = test_kaiser_wilhelm_instability_avoidance.__doc__
    S = initial(f"CO_KWI_UNSTABLE_{N}",
                N_electrons=N, NG=N,
                T = 200,
                push_mode=2,
                push_amplitude=0
                ).test_run()
    assert get_dominant_mode(S) > 1, plots(S, show_animation=True)





