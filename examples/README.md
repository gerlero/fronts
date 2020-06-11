# Examples

The example files are grouped in four different cases, each one in its own subdirectory. Here is a list of cases and files:


## ``powerlaw``

A simple case based on the introductory example included the main README file.

Contains:

* **``solve.py``**: solve the case with `fronts.solve()`.
* **``inverse.py``**: more examples of usage of `fronts.solve()` and of `fronts.inverse()`.
* **``D.py``**: plot _D_ for this case.


## ``1INFILTR``

The _1INFILTR_ test case from [Hydrus-1D](https://www.pc-progress.com/en/Default.aspx?hydrus-1d), in horizontal

Contains:

* **``solve.py``**: solve the case with `fronts.solve()`.
* **``validation.py``**: results for the same case obtained using Hydrus for comparison.
* **``D.py``**: plot _D_ for this case.


## ``HF135``

Infiltration into an HF135 nitrocellulose membrane. Data from the [PhD work of J. R. Buser](http://hdl.handle.net/1773/38064).

Contains:

* **``solve.py``**: solve the lateral flow case with `fronts.solve()`.
* **``refine.py``**: get a rough approximation of the solution to the lateral flow case using `fronts.solve()` with a high tolerance, and then refine it with both `fronts.solve()` and `fronts.solve_from_guess()`.
* **``radial.py``**: radial (cylindrical) flow case solved with `fronts.solve_flowrate()`.
* **``inverse1.py``**: use `fronts.inverse()` to extract _D_ from a solution. Here, the solution is obtained with 
`fronts.solve()`. The extracted _D_ is then used with `fronts.solve()` and the same conditions to verify that an equivalent solution is obtained.
* **``inverse2.py``**: use `fronts.inverse()` to obtain _D_ 
from the validation case and then use it to solve the same problem.
* **``validation.py``**: results with the same case solved with [porousMultiphaseFoam](https://github.com/phorgue/porousMultiphaseFoam) for comparison.
* **``D.py``**: plot _D_ for this case.


## ``exact``

Case with a _D_ function proposed by [J. R. Philip](https://doi.org/10.1071/PH600001) that has an exact solution.

Contains:

* **``solve.py``**: solve the case with `fronts.solve()` and compare with the exact solution.
* **``fromguess.py``**: solve the case with `fronts.solve_from_guess()` and compare with the exact solution.
* **``D.py``**: plot D for this case.
