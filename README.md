[<img alt="Fronts" src="https://raw.githubusercontent.com/gerlero/fronts/main/resources/logo.png" height="100">](https://github.com/gerlero/fronts)

Fronts is a Python numerical library for nonlinear diffusion problems based on the Boltzmann transformation.

```python
Python 3.9.6 (default, Sep 26 2022, 11:37:49)
>>> import fronts
>>> θ = fronts.solve(D="exp(7*θ)/2", i=0, b=1)  # i: initial value, b: boundary value
>>> θ(r=10, t=3) 
0.9169685387070694
>>> θ.d_dr(10,3)  # ∂θ/∂r
-0.01108790437249313
>>> print("Welcome to the Fronts project page.")
```

[![Documentation](https://img.shields.io/readthedocs/fronts)](https://fronts.readthedocs.io/)
[![GitHub Actions - CI](https://github.com/gerlero/fronts/workflows/CI/badge.svg)](https://github.com/gerlero/fronts/actions)
[![Code coverage](https://img.shields.io/codecov/c/gh/gerlero/fronts)](https://codecov.io/gh/gerlero/fronts)
[![PyPI](https://img.shields.io/pypi/v/fronts?color=%2300b0f0)](https://pypi.org/project/fronts/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fronts)](https://pypi.org/project/fronts/)
[![Docker image](https://img.shields.io/badge/docker%20image-microfluidica%2Ffronts-0085a0)](https://hub.docker.com/r/microfluidica/fronts)


| ⚡️ Fronts is also available [as a Julia package](https://github.com/gerlero/Fronts.jl). We recommend using the Julia version, particularly where performance is important |
| ---- |

## Overview

With Fronts, you can effortlessly find solutions to many problems of nonlinear diffusion along a semi-infinite axis **r**, i.e.:

$$\frac{\partial\theta}{\partial t} = \nabla\cdot\left[D(\theta)\frac{\partial\theta}{\partial r}\mathbf{\hat{r}}\right]$$

where _D_ is a known positive function and _θ_ is an unkown function of _r_ and _t_.

Fronts includes functionality to solve problems with a Dirichlet boundary condition (start with [``fronts.solve()``](https://fronts.readthedocs.io/en/stable/stubs/fronts.solve.html)), as well as some radial problems with a fixed-flowrate boundary condition (with [``fronts.solve_flowrate()``](https://fronts.readthedocs.io/en/stable/stubs/fronts.solve_flowrate.html)). In every case, _D_ can be any function defined by the user or obtained from the [``fronts.D``](https://fronts.readthedocs.io/en/stable) module. 

It works by transforming the above nonlinear partial differential equation (PDE) into a more manageable (but still nonlinear) ordinary differential equation (ODE), using a technique known as the [Boltzmann transformation](https://en.wikipedia.org/wiki/Boltzmann–Matano_analysis), which it then solves with a combination of high-order numerical ODE integration (provided by the [SciPy library](https://scipy.org/scipylib/index.html)) and specialized logic.

For this class of problems, you will find that Fronts can be easier to use, faster, and more robust than the classical numerical PDE solvers you would otherwise have to use. Moreover, the solutions found by Fronts are such that their partial derivatives and flux fields are also available in continuous form. Finally, a considerable effort has been made to have Fronts "just work" in practice, with no adjustment of numerical parameters required (in fact, the functions mentioned so far do not even require a starting mesh).

Fronts can also help you solve the inverse problem of finding _D_ when _θ_ is given. Every feature of Fronts is covered in the [documentation](https://fronts.readthedocs.io), and the project includes many [example cases](https://github.com/gerlero/fronts/tree/main/examples) (though you may start with the Usage section below).

Fronts is open source and works great with the tools of the [SciPy ecosystem](https://www.scipy.org/about.html).


## Why Fronts?

Problems compatible with Fronts appear in many areas of physics. For instance, if we take _θ_ as the water content or saturation and _D_ as the moisture diffusivity, the above equation translates into what is known as the moisture diffusivity equation, which is a special case of the [Richards equation](https://en.wikipedia.org/wiki/Richards_equation) that describes capillary flow in porous media. For this application, Fronts even includes implementations of the commonly used models: [``fronts.D.brooks_and_corey()``](https://fronts.readthedocs.io/en/stable/stubs/fronts.D.brooks_and_corey.html) and [``fronts.D.van_genuchten()``](https://fronts.readthedocs.io/en/stable/stubs/fronts.D.van_genuchten.html).

Of particular interest to the creators of Fronts is the fact that it can be used to model the configuration known as "lateral flow" in the field of paper-based microfluidics. The name "Fronts" is a reference to the wetting fronts that appear under these conditions, the study of which motivated the creation of this software.

Other problems of this class appear in the study of the diffusion of solutions in polymer matrices as well as diffusion problems in solids (e.g. annealing problems in metallurgy). 

As mentioned before, if your problem is supported, you can expect Fronts to be easier to use, faster, and more robust than other tools. Try it out!


## Installation

Fronts currently runs on Python 3.7 and later.

Install Fronts with [pip](https://pip.pypa.io/en/stable/) by running this command in a terminal:

```sh
python3 -m pip install fronts
```

This will download and install the [most recent version of Fronts available on PyPI](https://pypi.org/project/fronts/).

##### Optional: Matplotlib

Running the bundled examples requires the visualization library [Matplotlib](https://matplotlib.org). This library is not installed automatically with Fronts, so if you don't already have it, you may want to install it manually by running:

```sh
python3 -m pip install matplotlib
```

Optionally, Fronts can be installed in a [virtual environment](https://docs.python.org/3.8/tutorial/venv.html), or the ```--user```  option can be added to the previous commands to install the packages for the current user only (which does not require system administrator privileges).


## Usage

Let's say we want to solve the following initial-boundary value problem:

> Find _c_ such that:
>
> $$
  \begin{cases}
  \dfrac{\partial c}{\partial t} = \dfrac{\partial}{\partial r}\left(c^4\dfrac{\partial c}{\partial r}\right) & r>0,t>0\\
  c(r,0)=0.1 & r>0\\
  c(0,t)=1 & t>0\\
  \end{cases}
  $$

With Fronts, all it takes is a call to [``fronts.solve()``](https://fronts.readthedocs.io/en/stable/stubs/fronts.solve.html). The function requires the diffusivity function ``D``, which we pass as an expression so that ``solve()`` can get the derivatives it needs by itself (alternatively, in this case we could also have used [``fronts.D.power_law()``](https://fronts.readthedocs.io/en/stable/stubs/fronts.D.power_law.html#fronts.D.power_law.) to obtain `D`). Besides ``D``, we only need to pass the initial and boundary values as ``i`` and ``b``. The Python code is:

```python
import fronts
c = fronts.solve(D="c**4", i=0.1, b=1)
```

The call to ``solve()`` finishes within a fraction of a second. ``c`` is assigned a [``Solution``](https://fronts.readthedocs.io/en/stable/stubs/fronts.Solution.html) object, which can be called directly but also has some interesting methods such as [``d_dr()``](https://fronts.readthedocs.io/en/stable/stubs/fronts.Solution.html#fronts.Solution.d_dr), [``d_dt()``](https://fronts.readthedocs.io/en/stable/stubs/fronts.Solution.html#fronts.Solution.d_dt) and [``flux()``](https://fronts.readthedocs.io/en/stable/stubs/fronts.Solution.html#fronts.Solution.flux).

We can now plot the solution for arbitrary _r_ and _t_. For example, with _r_ between 0 and 10 and _t_=60:

```python
import numpy as np
import matplotlib.pyplot as plt

r = np.linspace(0, 10, 200)
plt.plot(r, c(r, t=60))
plt.xlabel("r")
plt.ylabel("c")
plt.show()
```

The plot looks like this:

<img alt="c plot" src="https://raw.githubusercontent.com/gerlero/fronts/main/resources/powerlaw_c.png" height=400>

Finally, let us plot the flux at _t_=60:

```python
plt.plot(r, c.flux(r, t=60))
plt.xlabel("r")
plt.ylabel("flux")
plt.show()
```

<img alt="flux plot" src="https://raw.githubusercontent.com/gerlero/fronts/main/resources/powerlaw_flux.png" height=400>


## Project links

* [**Documentation**](https://fronts.readthedocs.io)
* [**Examples**](https://github.com/gerlero/fronts/tree/main/examples)
* [**Changelog**](https://github.com/gerlero/fronts/blob/main/CHANGELOG.md)


## Authors

* **Gabriel S. Gerlero** [@gerlero](https://github.com/gerlero)
* **Pablo A. Kler** [@pabloakler](https://github.com/pabloakler)
* **Claudio L. A. Berli**

Fronts was conceived and is developed by members of the [Santa Fe Microfluidics Group (GSaM)](https://microfluidica.ar) at the [Research Center for Computational Methods (CIMEC, UNL-CONICET)](https://cimec.conicet.gov.ar) and the [Institute of Technological Development for the Chemical Industry (INTEC, UNL-CONICET)](https://intec.conicet.gov.ar) in Santa Fe, Argentina.

[<img alt="CIMEC (UNL-CONICET)" src="https://raw.githubusercontent.com/gerlero/fronts/main/resources/CIMEC.png" height=70>](https://cimec.conicet.gov.ar) &nbsp; [<img alt="INTEC (UNL-CONICET)" src="https://raw.githubusercontent.com/gerlero/fronts/main/resources/INTEC.png" height=70>](https://intec.conicet.gov.ar) &nbsp; [<img alt="GSaM" src="https://raw.githubusercontent.com/gerlero/fronts/main/resources/GSaMLogo.png" height=60>](https://microfluidica.ar)


## License

Fronts is open-source software available under the BSD 3-clause license.






