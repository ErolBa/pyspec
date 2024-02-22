#!/usr/bin/env python

import sys
from py_spec import SPECout
import matplotlib
matplotlib.use("Qt5Agg") # must come before importing plt
import matplotlib.pyplot as plt

num_args = len(sys.argv)

if(num_args<2):
    raise ValueError("Not enough input arguments!")
else:
    for n in range(1, num_args):
        
        fname = sys.argv[1]
        if(fname[-3:]=='.sp'):
            fname += '.h5'

        o = SPECout(fname)
        o.plot_poincare()

        plt.tight_layout()
plt.show()
