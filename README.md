# <img alt="Fronts" src="https://raw.githubusercontent.com/gerlero/fronts/master/resources/logo.png" height="100">

Fronts is a Python numerical library for solving one-dimensional transient nonlinear diffusion problems in semi-infinite domains.

Fronts finds solutions to initial-boundary value problems of the form:

> **General problem**
> 
> Given a scalar-valued positive function _D_, scalars _Si_, _Sb_ and _ob_, and coordinate unit vector **ȓ**, find a function _S_ of _r_ and _t_ such that:
> 
> <img alt="General problem" src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Bcases%7D%20%5Cdfrac%7B%5Cpartial%20S%7D%7B%5Cpartial%20t%7D%20%3D%20%5Cnabla%5Ccdot%5Cleft%5BD%5Cleft%28S%5Cright%29%5Cdfrac%7B%5Cpartial%20S%7D%7B%5Cpartial%20r%7D%5Cmathbf%7B%5Chat%7Br%7D%7D%5Cright%20%5D%20%26%20r%3Er_b%28t%29%2Ct%3E0%5C%5C%20S%28r%2C%200%29%20%3D%20S_i%20%26%20r%3E0%20%5C%5C%20S%28r_b%28t%29%2C%20t%29%20%3D%20S_b%20%26%20t%3E0%20%5C%5C%20r_b%28t%29%20%3D%20o_b%5Csqrt%20t%5Cend%7Bcases%7D">

Fronts works by transforming the governing nonlinear partial differential equation (PDE) into a more manageable (but still nonlinear) ordinary differential equation (ODE), using a technique known as the [Boltzmann transformation](https://en.wikipedia.org/wiki/Boltzmann–Matano_analysis), which it then solves with a combination of numerical ODE solvers and specialized logic.

For this class of problems, you will find that Fronts can be easier to use, faster, and more robust than the classical numerical PDE solvers you would otherwise have to use.

In some instances, Fronts can also solve the inverse problem of finding _D_ when _S_ is given. And, if you need something a little different, Fronts gives you easy access to the underlying ODE so that you can use your own solving algorithm or boundary condition (which you are then welcome to contribute to the project!).

Fronts is open source and works great with [NumPy](https://numpy.org) and [SciPy](https://www.scipy.org/scipylib/index.html). 


## Common problem

If the general problem supported by Fronts looks too complicated, note that in the common case where **ȓ** is a Cartesian unit vector and the boundary is fixed at _r_=0, the problem can be reduced to what we call the common problem:

> **Common problem**
> 
> Given a scalar-valued positive function _D_, and scalars _Si_ and _Sb_, find a function _S_ of _r_ and _t_ such that:
>
> <img alt="Common problem" src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Bcases%7D%20%5Cdfrac%7B%5Cpartial%20S%7D%7B%5Cpartial%20t%7D%20%3D%20%5Cdfrac%7B%5Cpartial%7D%7B%5Cpartial%20r%7D%20%5Cleft%28D%5Cleft%28S%5Cright%29%5Cdfrac%7B%5Cpartial%20S%7D%7B%5Cpartial%20r%7D%5Cright%29%20%26%20r%3E0%2Ct%3E0%5C%5C%20S%28r%2C%200%29%20%3D%20S_i%20%26%20r%3E0%20%5C%5C%20S%280%2C%20t%29%20%3D%20S_b%20%26%20t%3E0%20%5Cend%7Bcases%7D">


The main solver function ``solve()`` will assume that you want to work with this common problem unless you explicitly provide the optional `radial` and `ob` parameters. 


## Uses

Problems supported by Fronts appear in many areas of physics. For instance, if we take _S_ as the water content or saturation and _D_ as the moisture diffusivity, the above PDE translates into what is known as the moisture diffusivity equation, which is a special case of the [Richards equation](https://en.wikipedia.org/wiki/Richards_equation) that models fluid flow in unsaturated porous media.

Of particular interest to the creators of Fronts is the fact that the moisture diffusivity equation as supported by Fronts can directly describe the phenomenon known as lateral flow in the field of paper-based microfluidics. In fact, the name "Fronts" is a reference to the wetting fronts that appear under these conditions, the study of which motivated the creation of this library.

Other problems of this class appear in the study of diffusion of solutions in polymer matrices as well as diffusion problems in solids (e.g. annealing problems in metallurgy). 

As mentioned before, if your problem is supported, you can expect Fronts to be easier to use, faster, and more robust than other tools. Try it out!

## Installation

### Prerequisites

* **Python**. Fronts runs on Python 3.5 and later, as well as on the older Python 2.7. It has been tested on various releases of Python 2.7, 3.5, 3.6, 3.7 and 3.8.

* **pip**. Installation of Fronts requires the Python package manager [pip](https://pip.pypa.io/en/stable/) to be installed on your system.

### Installation

Install Fronts by running the following command:

```
$ pip install fronts
```

This will install the [most recent version of Fronts available on PyPI](https://pypi.org/project/fronts/).

##### Optional: Matplotlib

Running the bundled examples requires the visualization library [Matplotlib](https://matplotlib.org). This library is not installed automatically with Fronts, so if you don't already have it, you may want to install it manually by running:

```
$ pip install matplotlib
```


## Documentation and features

The following is a complete list of the functions and classes that Fronts provides, with a short description of each. You will find the full details on each object in the [API documentation](https://fronts.readthedocs.io).

### Solvers and solutions

* [**```fronts.solve()```**](https://fronts.readthedocs.io/en/latest/stubs/fronts.solve.html) — meshless solver

    ```solve``` solves any instance of the general problem. Returns a ```SemiInfiniteSolution```.
    
* [**```fronts.solve_from_guess()```**](https://fronts.readthedocs.io/en/latest/stubs/fronts.solve_from_guess.html) — mesh-based solver
    
    ```solve_from_guess``` works like ``solve`` but it uses a different procedure that starts from a guess of the solution on an initial mesh. It supports the same kind of problems than ```solve```. Although usually faster, ```solve_from_guess``` is significantly less robust than `solve`—whether it converges will usually depend heavily on the problem, the initial mesh and the guess of the solution. It also returns a ```SemiInfiniteSolution``` on success.


* [**```fronts.Solution```**](https://fronts.readthedocs.io/en/latest/stubs/fronts.Solution.html), [**```fronts.SemiInfiniteSolution```**](https://fronts.readthedocs.io/en/latest/stubs/fronts.SemiInfiniteSolution.html) — continuous solutions

    ```Solution``` objects provide the continuous functions ```S```, ```dS_dr```, ```dS_dt``` and ```flux``` that build up the solution to a problem. The solvers in Fronts return a ```SemiInfiniteSolution``` (a subclass of ```Solution```) as part of their results. If you called ```ode``` and solved the ODE yourself, you can create a ```Solution``` or ```SemiInfiniteSolution``` by passing the solution to the ODE to the appropiate constructor.
    
    Note that in problems of the moisture diffusivity equation or horizontal Richards equation, the diffusive flux (which can be obtained by calling ```flux``` on a ```Solution``` object) gives the velocity of the wetting fluid. In particular, if `S` is taken to mean volumetric water content, it is the Darcy velocity; if `S` is saturation, it is the fluid's true velocity. These velocity fields can be used directly in more complex problems of solute transport.


* [**```fronts.inverse()```**](https://fronts.readthedocs.io/en/latest/stubs/fronts.inverse.html) — solve the inverse problem
    
    ```inverse``` solves the inverse problem of finding _D_ when _S_ is known. For instance, ```inverse``` can extract _D_ from experimental results. The returned _D_ function can be used in Fronts to solve other problems.
    

### Boltzmann transformation and ODE

* [**```fronts.o()```**](https://fronts.readthedocs.io/en/latest/stubs/fronts.o.html), [**```fronts.do_dr()```**](https://fronts.readthedocs.io/en/latest/stubs/fronts.do_dr.html), [**```fronts.do_dt()```**](https://fronts.readthedocs.io/en/latest/stubs/fronts.do_dt.html), [**```fronts.r()```**](https://fronts.readthedocs.io/en/latest/stubs/fronts.r.html), [**```fronts.t()```**](https://fronts.readthedocs.io/en/latest/stubs/fronts.t.html), [**```fronts.as_o()```**](https://fronts.readthedocs.io/en/latest/stubs/fronts.as_o.html) — Boltzmann transformation

    These are convenience functions for working with the Boltzmann transformation.

* [**```fronts.ode()```**](https://fronts.readthedocs.io/en/latest/stubs/fronts.ode.html) — access the ODE

    The ```ode``` function transforms the PDE into its corresponding ODE using the Boltzmann transformation. ```ode``` returns _fun_ and _jac_ callables that are directly compatible with SciPy's solvers (i.e., those in the  [```scipy.integrate```](https://docs.scipy.org/doc/scipy/reference/integrate.html) module). The solvers in Fronts actually use this function internally. You may call this function if you want to solve the ODE yourself instead of using Fronts' solvers, for example if you need to deal with a different boundary condition or want to use your own solving algorithm.

### _D_ functions and ```fronts.D```

Many of the functions in Fronts either take or return _D_ functions to work. _D_ functions have to be defined as follows:

> ``D`` : _callable_
> 
> Twice-differentiable function that maps the range of _S_ to positive values. It can be called as ``D(S)`` to evaluate it at `S`. It can also be called as ``D(S, derivatives)`` with `derivatives` equal to 1 or 2, in which case the first `derivatives` derivatives of the function evaluated at the same `S` are included (in order) as additional return values. While mathematically a scalar function, `D` operates in a vectorized fashion with the same semantics when `S` is a `numpy.ndarray`.
 

With the above definition you can easily write any functions you need to solve your particular problems. 

Fronts also comes with a submodule ```fronts.D``` that lets you access some predefined functions:

* [**```fronts.D.constant()```**](https://fronts.readthedocs.io/en/latest/stubs/fronts.D.constant.html) — create a constant function:

    <img src="https://latex.codecogs.com/svg.latex?%5Csmall%20D%28S%29%3DD">

* [**```fronts.D.power_law()```**](https://fronts.readthedocs.io/en/latest/stubs/fronts.D.power_law.html) — create a function of the form:

    <img src="https://latex.codecogs.com/svg.latex?%5Csmall%20D%28S%29%3Da%20S%5Ek%20&plus;%20%5Cvarepsilon">

* [**```fronts.D.van_genuchten()```**](https://fronts.readthedocs.io/en/latest/stubs/fronts.D.van_genuchten.html) — create a [Van Genuchten](https://doi.org/10.2136/sssaj1980.03615995004400050002x) moisture diffusivity function:

    <img src="https://latex.codecogs.com/svg.latex?%5Csmall%20D%28S%29%3D%5Cfrac%7B%281-m%29K_s%7D%7B%5Calpha%20m%20%28S_s-S_r%29%7DS_e%5E%7B%28l-%5Cfrac%7B1%7D%7Bm%7D%29%7D%5Cleft%28%281-S_e%5E%5Cfrac%7B1%7D%7Bm%7D%29%5E%7B-m%7D%20&plus;%20%281-S_e%5E%5Cfrac%7B1%7D%7Bm%7D%29%5Em%20-%202%20%5Cright%29">
    
    where _S_ is either water content or saturation, and _Se_ is defined as:
    
    <img src="https://latex.codecogs.com/svg.latex?%5Csmall%20S_e%20%3D%20%5Cfrac%7BS-S_r%7D%7BS_s-S_r%7D">


* [**```fronts.D.richards()```**](https://fronts.readthedocs.io/en/latest/stubs/fronts.D.richards.html) — make a moisture diffusivity function from the hydraulic conductivity function _K_ and the capillary capacity function _C_ using the definition: 
    
    <img src="https://latex.codecogs.com/svg.latex?%5Csmall%20D%28S%29%3D%5Cfrac%7BK%28S%29%7D%7BC%28S%29%7D">

    
## Examples

### Introductory example

_Plotting the solution in this example requires_ [Matplotlib](https://matplotlib.org)_._

Let us solve the following initial-boundary value problem defined in a semi-infinite domain:

> **Example problem**
>
> Find _S_ such that:
>
> <img alt="Example problem" src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Bcases%7D%20%5Cdfrac%7B%5Cpartial%20S%7D%7B%5Cpartial%20t%7D%20%3D%20%5Cdfrac%7B%5Cpartial%7D%7B%5Cpartial%20r%7D%5Cleft%28S%5E4%5Cdfrac%7B%5Cpartial%20S%7D%7B%5Cpartial%20r%7D%5Cright%29%20%26%20r%3E0%2Ct%3E0%20%5C%5C%20S%28r%2C0%29%20%3D%200.1%20%26%20r%3E0%20%5C%5C%20S%280%2Ct%29%20%3D%201%20%26%20t%3E0%20%5C%5C%20%5Cend%7Bcases%7D">

By comparing the example problem with the common problem introduced above, we see that  the parameters are:

<img alt="Parameters" src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Bcases%7D%20D%28S%29%20%3D%20S%5E4%20%5C%5C%20S_i%20%3D%200.1%20%5C%5C%20S_b%20%3D%201%20%5Cend%7Bcases%7D">

In this case it is not necessary to write the function `D` it ourselves. The function we need can be obtained from the ``fronts.D`` module:

```python
from fronts.D import power_law
D = power_law(k=4)
```

We are now ready to solve the problem with ``fronts.solve``. We simply pass it the parameters ``D``, ``Si`` and ``Sb``.

```python
from fronts import solve
solution = solve(D, Si=0.1, Sb=1)
```

The call to ```fronts.solve``` completes within a second and we get back a ```SemiInfiniteSolution``` object, which holds the functions ```S```, ```dS_dr```, ```dS_dt```and ```flux```.

We can now plot _S_ for arbitrary _r_ and _t_. For example, with _r_ between 0 and 10 and _t_=60:

```python
import matplotlib.pyplot as plt
r = np.linspace(0, 10, 200)
plt.plot(r, solution.S(r, t=60))
plt.xlabel("r")
plt.ylabel("S")
plt.show()
```

The plot will look like this:

<img alt="S plot" src="https://raw.githubusercontent.com/gerlero/fronts/master/resources/powerlaw_S.png" height=400>

Finally, let us plot the flux at _t_=60:

```python
plt.plot(r, solution.flux(r, t=60))
plt.xlabel("r")
plt.ylabel("flux")
plt.show()
```

<img alt="flux plot" src="https://raw.githubusercontent.com/gerlero/fronts/master/resources/powerlaw_flux.png" height=400>

### More examples

The included examples can be found in the ``examples`` directory of this project. The directory contains the following files:


* subdirectory **``powerlaw/``** — cases based on the introductory example presented above
    * **``solve.py``**: solve the case with `fronts.solve()`.
    * **``radial.py``**: solve a radial case (with a moving boundary) using `fronts.solve()`.
    * **``inverse.py``**: more examples of usage of `fronts.solve()` and of`fronts.inverse()`.

* subdirectory **``1INFILTR/``** — the _1INFILTR_ test case from [Hydrus-1D](https://www.pc-progress.com/en/Default.aspx?hydrus-1d), in horizontal
    * **``solve.py``**: solve the case with `fronts.solve()`.
    * **``validation.py``**: results for the same case obtained using Hydrus for comparison.
* subdirectory **``HF135/``**— lateral flow case in an HF135 nitrocellulose membrane (data from the [PhD work of J.R. Buser](http://hdl.handle.net/1773/38064))
    * **``solve.py``**: solve the case with `fronts.solve()`.
    * **``refine.py``**: get a rough approximation of the solution using `fronts.solve()` with a high tolerance, and then refine it with both `fronts.solve()` and `fronts.solve_from_guess()`.
    * 🐌 **``inverse1.py``**: use `fronts.inverse()` to extract _D_ from a solution. Here, the solution is obtained with 
`fronts.solve()`. The extracted _D_ is then used with `fronts.solve()` and the
same conditions to verify that an equivalent solution is obtained.
    * 🐌 **``inverse2.py``**: use `fronts.inverse()` to obtain _D_ 
from the validation case and then use it to solve the same problem. 
    * **``validation.py``**: results with the same case solved with [porousMultiphaseFoam](https://github.com/phorgue/porousMultiphaseFoam) for comparison.
* subdirectory **``exact/``** — solve a case with a _D_ function proposed by [Philip](https://doi.org/10.1071/PH600001) that has an exact solution
    * **``solve.py``**: solve the case with `fronts.solve()` and compare with the exact solution.
    * **``fromguess.py``**: solve the case with `fronts.solve_from_guess()` and compare with the exact solution.


**Note:** the examples marked with 🐌 are significantly more computationally intensive and may take more than a minute to run to completion. All other cases should finish within a few seconds at the most.

## Authors

* **Gabriel S. Gerlero** [@gerlero](https://github.com/gerlero)
* **Pablo A. Kler** [@pabloakler](https://github.com/pabloakler)
* **Claudio L.A. Berli**

Fronts was conceived and is developed by members of the [Santa Fe Microfluidics Group (GSaM)](http://www.microfluidica.com.ar) at the [Research Center for Computational Methods (CIMEC, UNL-CONICET)](https://www.cimec.org.ar) and the [Institute of Technological Development for the Chemical Industry (INTEC, UNL-CONICET)](https://intec.conicet.gov.ar) in Santa Fe, Argentina.



<img alt="CIMEC (UNL-CONICET)" src="https://raw.githubusercontent.com/gerlero/fronts/master/resources/CIMEC.png" height=70> &nbsp; <img alt="INTEC (UNL-CONICET)" src="https://raw.githubusercontent.com/gerlero/fronts/master/resources/INTEC.png" height=65> &nbsp; <img alt="GSaM" src="https://raw.githubusercontent.com/gerlero/fronts/master/resources/GSaMLogo.png" height=65> 

 


## License

Fronts is open-source software available under the BSD 3-clause license.






