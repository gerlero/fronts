#!/usr/bin/env python

"""
1INFILTR case from Hydrus-1D in horizontal orientation.

Data obtained using Hydrus-1D version 4.17.0140
1001 nodes, water content tolerance = 1e-6
"""

from __future__ import division, absolute_import, print_function

import sys
import os

import numpy as np
import matplotlib.pyplot as plt


r_unit = "cm"
t_unit = "h"

_filename = os.path.join(sys.path[0], "Nod_Inf.out")


t = []
r = None
theta = []
velocity = []

with open(_filename) as file:
    for i, line in enumerate(file):
        if i <= 10: continue  # Skip time 0
        if "Time:" in line:
            for s in line.split():
                try:
                    found = float(s)
                    break
                except ValueError:
                    pass

            t.append(found)

            r_, theta_, velocity_ = np.loadtxt(_filename,
                                               skiprows=i+5,
                                               max_rows=1001,
                                               usecols=(1, 3, 6),
                                               unpack=True)

            if r is None:
                r = -r_
            else:
                assert np.all(r == -r_)

            theta.append(theta_)
            velocity.append(-velocity_)

t = np.asarray(t)
r = np.asarray(r)
theta = np.asarray(theta)
velocity = np.asarray(velocity)

assert(np.ndim(t) == 1)
assert(np.ndim(r) == 1)
assert(np.ndim(theta) == 2)
assert(np.ndim(velocity) == 2)


name = "Hydrus-1D"

if __name__ == '__main__':

    fig = plt.figure()
    fig.canvas.set_window_title("Water content plot")

    plt.title("Water content fields")
    for t_, theta_ in zip(t, theta):
        plt.plot(r, theta_, label="{}, t={} {}".format(name, t_, t_unit))
    plt.xlabel("r [{}]".format(r_unit))
    plt.ylabel("water content [-]")
    plt.grid(which='both')
    plt.legend()

    fig = plt.figure()
    fig.canvas.set_window_title("Velocity plot")

    plt.title("Velocity fields")
    for t_, velocity_ in zip(t, velocity):
        plt.plot(r, velocity_, label="{}, t={} {}".format(name, t_, t_unit))
    plt.xlabel("r [{}]".format(r_unit))
    plt.ylabel("Darcy velocity [{}/{}]".format(r_unit, t_unit))
    plt.grid(which='both')
    plt.legend()

    plt.show()
