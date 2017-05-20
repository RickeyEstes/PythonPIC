# coding=utf-8
from pythonpic.configs import run_coldplasma, run_beamplasma, run_wave, run_twostream

for conf in run_wave, run_coldplasma, run_twostream, run_beamplasma:
    conf.main()