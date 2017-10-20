# coding=utf-8
import os
from time import time

import numpy as np
import pytest

from ..configs.run_coldplasma import initial
from ..helper_functions import helpers
from ..visualization import animation
from ..visualization.static_plots import static_plots


@pytest.fixture(scope="module")
def helper_short_simulation():
    if "DISPLAY" not in os.environ.keys():
        print("Not running display test right now.")
        return False
    else:
        run_name = "visualization_test"
        S = initial(run_name, save_data=False).run().postprocess()
        return S


def test_static_plots(helper_short_simulation):
    S = helper_short_simulation
    if S:
        static_plots(S, S.filename.replace(".hdf5", ".png"))
        assert True

# def test_animation(helper_short_simulation):
#     S = helper_short_simulation
#     if S:
#         Animation.OneDimAnimation(S).full_animation(True)
#         assert True


def test_writer_manual_speed(helper_short_simulation):
    S = helper_short_simulation
    if S:
        start_time = time()
        frames = list(np.arange(0, S.NT,
                                helpers.calculate_particle_iter_step(S.NT),
                                dtype=int)[::10])
        animation.OneDimAnimation(S).snapshot_animation()
        endtime = time()
        runtime = endtime - start_time
        print(runtime)
        assert runtime

# @pytest.mark.parametrize("writer", ['ffmpeg', 'ffmpeg_file', 'mencoder'])
# def test_writer_speed(helper_short_simulation, writer):
#     S = helper_short_simulation
#     start_time = time()
#     Animation(S, save=True, writer=writer)
#     endtime = time()
#     runtime = endtime - start_time
#     print(writer, runtime)
#     assert runtime
