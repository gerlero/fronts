Fronts
======

Welcome to the reference documentation for Fronts. This documentation covers the usage of all available functions and classes.

For an introduction to the software, please refer to the README file, which is displayed on the project's `GitHub <http://github.com/gerlero/fronts>`_ and `PyPI <https://pypi.org/project/fronts>`_ pages. Users may also want to review the example cases, found on the GitHub page under the *examples* directory.

.. currentmodule:: fronts

Main module ``fronts``
----------------------


Solvers
~~~~~~~

.. autosummary::
	:toctree: stubs/
	   
	solve
	solve_from_guess  
	inverse

Solutions
~~~~~~~~~

.. autosummary::
	:toctree: stubs/
	:nosignatures:
	   
	SemiInfiniteSolution
	Solution


Boltzmann transformation
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
	:toctree: stubs/
	   
	ode
	o
	do_dr
	do_dt
	r
	t
	as_o


Submodule ``fronts.D``: included D functions
--------------------------------------------

.. autosummary::
	:toctree: stubs/
	
	D.constant
	D.power_law
	D.van_genuchten
	D.richards


* :ref:`genindex`
* :ref:`search`
