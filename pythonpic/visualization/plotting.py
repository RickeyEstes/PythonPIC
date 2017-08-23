"""plotting suite for simulation data analysis, can be called from command line"""
# coding=utf-8
import argparse
import os


from ..classes import simulation
from ..visualization import static_plots



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="hdf5 file name for storing data")
    parser.add_argument('-save', action='store_false')
    parser.add_argument('-lines', action='store_true')
    args = parser.parse_args()
    if args.filename[-5:] != ".hdf5":
        args.filename += ".hdf5"

    plots(args.filename, True, False, True, False)
