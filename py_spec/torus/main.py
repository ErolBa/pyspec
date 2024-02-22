# Routines for running SPEC in a tokamak geometry

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
from scipy import integrate, interpolate, optimize
import subprocess
import numpy.linalg as linalg
import numba
import os
import contextlib
import sympy as sym
from scipy.optimize import minimize_scalar
from rich.console import Console

from .input_dict import input_dict

class SPECtorus():
    """
    Class for running the slab (G=1) case, including plotting, analyzing, and perturbing equilibrium
    """

    def hello():
        print("Hello from SPECtorus")
        
    