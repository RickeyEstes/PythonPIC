# coding=utf-8
import pytest
import numpy as np

from . import on_failure
from pythonpic.helper_functions.physics import did_it_thermalize
from ..run_twostream import initial


@pytest.mark.parametrize(["L", "NG", "N_electrons"], [
    (2*np.pi/100, 32, 512),
    (2*np.pi/200, 32, 512),
    # (200, 5000),
    # (400, 10000),
    ])
def test_linear_regime_beam_stability(L, NG, N_electrons):
    """Tests the simulation's behavior in modes expected to be linear."""
    run_name = f"TS_LINEAR_{L}_{NG}_{N_electrons}"
    dx = L / NG
    c = 1
    dt = dx / c
    T = 50000 * dt
    # NT = T / dt = 50000 * c / dx = 50000 * c * NG / L
    S = initial(run_name,
                NG=NG,
                L=L,
                T = T,
                N_electrons=N_electrons,
                v0 = 0.01,
                ).test_run()
    assert (~did_it_thermalize(S)).all(), ("A linear regime run came out "
                                           "unstable.", S.plots(*on_failure))



# TODO: redo this test
# @pytest.mark.parametrize(["L", "NG", "N_electrons"], [
#     (2 * np.pi * 1000, 32, 512),
#     (2 * np.pi * 2000, 32, 512),
#     ])
# def test_nonlinear_regime_beam_instability(L, NG, N_electrons):
#     run_name = f"TS_NONLINEAR_{L}_{NG}_{N_electrons}"
#     dx = L / NG
#     c = 1
#     dt = dx / c
#     T = 50000 * dt
#     S = initial(run_name,
#                                NG=NG,
#                                N_electrons=N_electrons,
#                                L=L,
#                                v0 = 0.01,
#                                T=T,
#                                ).test_run()
#     assert did_it_thermalize(S).all(), ("A nonlinear regime run came out stable.", plots(S, *on_failure))

# @pytest.mark.parametrize(["v0", "NT"], [
#     (0.1, 450),
#     (0.2, 450),
#     (0.3, 300),
#     # TEST: this needs an xfail
#     ])
# def test_electron_positron(v0, NT):
#     """the electron-positron run is much noisier
#     the particles do not really seem to jump between beams """
#     S = initial("TS_EP", NG=64, N_electrons=512, plasma_frequency=5, NT=NT, v0=v0, species_2_sign=-1,
#                                save_data=False)
#     average_velocities = [sp.velocity_history[:, int(sp.N / 2), 0].mean() for sp in S.list_species]
#     avg_velocity_difference = abs(average_velocities[1] - average_velocities[0])
#     print(avg_velocity_difference)
#     assert avg_velocity_difference > v0, plots(S, show=show_on_fail, save=False, animate=True)


# @pytest.mark.parametrize(["push_amplitude"], [
#    (0.95,), (1.05,), (1.25,),
#    ])
# def test_push_amplitude(push_amplitude):
#     S = initial(f"TS_PUSH_{push_amplitude}",
#                            NG=64,
#                            N_electrons=512,
#                            push_amplitude=push_amplitude,
#                            )
#
#     # TEST: finish this
def test_finish():
    """Tests whether the simulation finishes at all."""
    S = initial("TS_FINISH",
                NG=512,
                N_electrons=4096,
                plasma_frequency=0.05 / 4,
                ).test_run()
    assert True, ("The simulation did not finish.", S.plots(*on_failure))
