Fronts
======

Welcome to the reference documentation for Fronts!

Don't forget to read the README and check out the examples on the project's `GitHub page <http://github.com/gerlero/fronts>`_.

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
