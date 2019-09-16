#!/usr/bin/env python

"""
Lateral flow in a HF135 nitrocellulose membrane.

Solution to the case obtained with the porousMultiphaseFoam toolbox in OpenFOAM
(https://github.com/phorgue/porousMultiphaseFoam). The groundwaterFoam solver
was used.

"""

from __future__ import division, absolute_import, print_function

import sys
import os

import numpy as np
import matplotlib.pyplot as plt

_filename = os.path.join(sys.path[0], "groundwaterFoam_results.csv")

name = "porousMultiphaseFoam"

r, S, velocity = np.loadtxt(_filename, delimiter=',', skiprows=1, 
							usecols=(4, 3, 0), unpack=True)

t = 60

if __name__ == '__main__':

	plt.title("Solution at t={}".format(t))
	plt.plot(r, S, color='sandybrown', label=name)
	plt.xlabel("r")
	plt.ylabel("saturation")
	plt.grid(which='both')
	plt.legend()
	plt.show()

	plt.title("Solution at t={}".format(t))
	plt.plot(r, velocity, color='sandybrown', label=name)
	plt.xlabel("r")
	plt.ylabel("velocity")
	plt.grid(which='both')
	plt.legend()
	plt.show()

