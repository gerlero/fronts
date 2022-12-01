#!/usr/bin/env python3

"""
1INFILTR case from Hydrus-1D in horizontal orientation.

Data obtained using Hydrus-1D version 4.17.0140
1001 nodes, water content tolerance = 1e-6
"""

import sys
import os
import itertools

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

            skip_lines = 5
            max_rows = 1001

            data = itertools.islice(file, skip_lines, max_rows+skip_lines)

            r_, theta_, velocity_ = np.loadtxt(data,
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
    fig.canvas.manager.set_window_title("Water content plot")

    plt.title("Water content fields")
    for t_, theta_ in zip(t, theta):
        plt.plot(r, theta_, label=f"{name}, t={t_} {t_unit}")
    plt.xlabel(f"r [{r_unit}]")
    plt.ylabel("water content [-]")
    plt.grid(which='both')
    plt.legend()

    fig = plt.figure()
    fig.canvas.manager.set_window_title("Velocity plot")

    plt.title("Velocity fields")
    for t_, velocity_ in zip(t, velocity):
        plt.plot(r, velocity_, label=f"{name}, t={t_} {t_unit}")
    plt.xlabel(f"r [{r_unit}]")
    plt.ylabel(f"Darcy velocity [{r_unit}/{t_unit}]")
    plt.grid(which='both')
    plt.legend()

    plt.show()
