# coding=utf-8
from pythonpic import plotting_parser
from pythonpic.helper_functions.physics import did_it_thermalize
from pythonpic.configs.run_twostream import two_stream_instability, plots


args = plotting_parser("Two stream instability")

S = two_stream_instability("TS_STABLE",
                           v0 = 0.01,
                           N_electrons=5000,
                           plasma_frequency=0.001,
                           T = 6000,
                           ).lazy_run()
print(did_it_thermalize(S))
plots(S, *args)

S = two_stream_instability("TS_UNSTABLE",
                           v0 = 0.01,
                           N_electrons=5000,
                           plasma_frequency=0.1,
                           T = 6000,
                           ).lazy_run()
print(did_it_thermalize(S))
plots(S, *args)

S = two_stream_instability("TSe-1",
                       NG=512,
                       N_electrons=4096,
                       plasma_frequency=0.05 / 4,
                       v0 = 1e-1
                       ).lazy_run()
print(did_it_thermalize(S))
plots(S, *args)
S = two_stream_instability("TSe-2",
                       NG=512,
                       N_electrons=4096,
                       plasma_frequency=0.05 / 4,
                       v0 = 1e-2,
                       ).lazy_run()
print(did_it_thermalize(S))
plots(S, *args)
S = two_stream_instability("TSe-3",
                       NG=512,
                       N_electrons=4096,
                       plasma_frequency=0.05 / 4,
                       v0 = 1e-3,
                       ).lazy_run()
print(did_it_thermalize(S))
plots(S, *args)
S = two_stream_instability("TSe-4",
                           NG=512,
                           N_electrons=4096,
                           plasma_frequency=0.05 / 4,
                           v0 = 1e-4,
                           ).lazy_run()
print(did_it_thermalize(S))
plots(S, *args)
S = two_stream_instability("TS90p",
                           NG=512,
                           N_electrons=4096,
                           plasma_frequency=0.05 / 4,
                           v0 = 0.9,
                           ).lazy_run()
print(did_it_thermalize(S))
plots(S, *args)
S = two_stream_instability("TSRANDOM1",
                       NG=512,
                       N_electrons=4096,
                       vrandom=1e-1,
                       ).lazy_run()
print(did_it_thermalize(S))
plots(S, *args)
S = two_stream_instability("TSRANDOM2",
                       NG=512, N_electrons=4096,
                       vrandom=1e-1).lazy_run()
print(did_it_thermalize(S))
plots(S, *args)