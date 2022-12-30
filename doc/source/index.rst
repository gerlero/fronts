Fronts
======

Welcome to the reference documentation for `Fronts <http://github.com/gerlero/fronts>`_. This documentation covers the usage of all available functions and classes.

For an introduction to the software, please refer to the README file, which is displayed on the project's `GitHub <http://github.com/gerlero/fronts>`_ and `PyPI <https://pypi.org/project/fronts>`_ pages.

Users may also want to look at the example cases, available on the GitHub page under the `examples <https://github.com/gerlero/fronts/tree/main/examples>`_ directory.

.. note::
   Documentation for the `Julia version of Fronts <https://github.com/gerlero/Fronts.jl>`_ is `available separately <https://gerlero.github.io/Fronts.jl/stable/>`_.  


.. currentmodule:: fronts

Main package ``fronts``
-----------------------


Solvers
~~~~~~~

.. autosummary::
    :toctree: stubs/
       
    solve
    solve_flowrate
    solve_from_guess  
    inverse

Solutions
~~~~~~~~~

.. autosummary::
    :toctree: stubs/
    :nosignatures:
       
    Solution
    BaseSolution


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


Module ``fronts.D``: Diffusivity functions
------------------------------------------

.. autosummary::
    :toctree: stubs/
    
    D.constant
    D.power_law
    D.brooks_and_corey
    D.van_genuchten
    D.letxs
    D.letd
    D.from_expr
    D.richards


* :ref:`genindex`
* :ref:`search`
