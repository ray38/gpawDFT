.. _advancedpoisson:

========================
Advanced Poisson solvers
========================

The ``PoissonSolver`` with default parameters uses zero boundary conditions on
the cell boundaries. This becomes a problem in systems involving large dipole
moment, for example (due to, e.g., plasmonic charge oscillation on a
nanoparticle). The potential due to the dipole is long-ranged and, thus, the
converged potential requires large vacuum sizes.

However, in LCAO approach large vacuum size is often unnecessary. Thus, to
avoid using large vacuum sizes but get converged potential, one can use two
approaches or their combination: 1) use multipole moment corrections or 2)
solve Poisson equation on a extended grid. These two approaches are
implemented in ``ExtendedPoissonSolver``. Also regular PoissonSolver in GPAW
has the option remove_moment.

In any nano-particle plasmonics calculation, it is necessary to use multipole
correction. Without corrections more than 10Å of vacuum is required for
converged results.


Multipole moment corrections
----------------------------

The boundary conditions can be improved by adding multipole moment
corrections to the density so that the corresponding multipoles of the
density vanish. The potential of these corrections is added to the obtained
potential. For a description of the method, see [#Castro2003]_.

This can be accomplished by following solver::

  from gpaw.poisson_extended import ExtendedPoissonSolver
  poissonsolver = ExtendedPoissonSolver(eps=eps,
                                        moment_corrections=4)

This corrects the 4 first multipole moments, i.e., `s`, `p_x`, `p_y`, and
`p_z` type multipoles. The range of multipoles can be changed by changing
``moment_corrections`` parameter. For example, ``moment_correction=9``
includes in addition to the previous multipoles, also `d_{xx}`, `d_{xy}`,
`d_{yy}`, `d_{yz}`, and `d_{zz}` type multipoles.

This setting suffices usually for spherical-like metallic nanoparticles, but
more complex geometries require inclusion of very high multipoles or,
alternatively, a multicenter multipole approach. For this, consider the
advanced syntax of the moment_corrections. The previous code snippet is
equivalent to::

  from gpaw.poisson_extended import ExtendedPoissonSolver
  poissonsolver = ExtendedPoissonSolver(eps=eps,
                                        moment_corrections=[{'moms': range(4), 'center': None}])

Here ``moment_corrections`` is a list of dictionaries with following
keywords: ``moms`` specifies the considered multipole moments, e.g.,
``range(4)`` equals to `s`, `p_x`, `p_y`, and `p_z` multipoles, and
``center`` specifies the center of the added corrections in atomic units
(``None`` corresponds to the center of the cell).

As an example, consider metallic nanoparticle dimer where the nanoparticle
centers are at ``(x1, y1, z1)`` Å and ``(x2, y2, z2)`` Å. In this case, the
following settings for the ``ExtendedPoissonSolver`` may be tried out::

  import numpy as np
  from ase.units import Bohr
  from gpaw.poisson_extended import ExtendedPoissonSolver
  moms = range(4)
  center1 = np.array([x1, y1, z1]) / Bohr
  center2 = np.array([x2, y2, z2]) / Bohr
  poissonsolver = ExtendedPoissonSolver(eps=eps,
                                        moment_corrections=[{'moms': moms, 'center': center1},
                                                            {'moms': moms, 'center': center2}])

When multiple centers are used, the multipole moments are calculated on
non-overlapping regions of the calculation cell. Each point in space is
associated to its closest center. See `Voronoi diagrams
<http://en.wikipedia.org/wiki/Voronoi_diagram>`_ for analogous illustration of
the partitioning of a plane.

.. [#Castro2003]
   A. Castro, A. Rubio, and M. J. Stott,
   Solution of Poisson's equation for finite systems using plane-wave methods,
   *Can. J. Phys.* **81**, 1151 (2003).
   `doi:10.1139/p03-078 <http://dx.doi.org/10.1139/p03-078>`_


Extended Poisson grid
---------------------

The multipole correction scheme is not always successful for complex system
geometries. For these cases, one can use a separate large grid just for
solving the Hartree potential. Such extended grid can be set up as follows::

  from gpaw.poisson_extended import ExtendedPoissonSolver
  poissonsolver = ExtendedPoissonSolver(eps=eps,
                                        extended={'gpts': (128, 128, 128),
                                                  'useprev': False})

This solves the Poisson equation on an extended grid. The size of the grid is
given **in units of coarse grid**. Thus, in this case the fine extended grid
used in evaluating the Hartree potential is of size (256, 256, 256). It is
important to **use grid sizes that are divisible by high powers of 2 to
accelerate the multigrid scheme** used in ``PoissonSolver``.

The ``useprev`` parameter describes whether the ``ExtendedPoissonSolver``
uses the previous solution as a initial potential for subsequent calls of the
function ``solver.solve()``. It is often reasonable to use ``useprev =
True``, which mimics the behaviour of the usual ``PoissonSolver`` in most
cases. In such cases, the value ``useprev = False`` would lead to significant
performance decrease since the Hartree potential is always calculated from
scratch.

.. note::

   When extended grid is in use, the implementation neglects the ``phi``
   parameter given to the ``solver.solve()`` due to its incompatibility
   with the extended grid.
