#!/usr/bin/env python

"""
Lateral flow in a HF135 nitrocellulose membrane.

Solution to the case obtained with the porousMultiphaseFoam toolbox (version
1906) for OpenFOAM (https://github.com/phorgue/porousMultiphaseFoam). The
groundwaterFoam solver was used.

"""

from __future__ import division, absolute_import, print_function

import sys
import os

import numpy as np
import matplotlib.pyplot as plt

_filename = os.path.join(sys.path[0], "groundwaterFoam_results.csv")

name = "porousMultiphaseFoam"

r_unit = "m"
t_unit = "s"

r, theta, velocity = np.loadtxt(_filename, delimiter=',', skiprows=1,
                                usecols=(4, 3, 0), unpack=True)

t = 60


if __name__ == '__main__':

    fig = plt.figure()
    fig.canvas.manager.set_window_title("Water content plot")

    plt.title("Saturation field at t={} {}".format(t, t_unit))
    plt.plot(r, theta, color='sandybrown', label=name)
    plt.xlabel("position [{}]".format(r_unit))
    plt.ylabel("water content [-]")
    plt.grid(which='both')
    plt.legend()

    fig = plt.figure()
    fig.canvas.manager.set_window_title("Velocity plot")

    plt.title("Velocity field at t={} {}".format(t, t_unit))
    plt.plot(r, velocity, color='sandybrown', label=name)
    plt.xlabel("position [{}]".format(r_unit))
    plt.ylabel("Darcy velocity [{}/{}]".format(r_unit, t_unit))
    plt.grid(which='both')
    plt.legend()

    plt.show()
