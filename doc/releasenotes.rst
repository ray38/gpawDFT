.. _releasenotes:

=============
Release notes
=============


Git master branch
=================

:git:`master <>`.

* New file-format for gpw-files.  Reading of old files should still work.
  Look inside the new files with::

      $ python -m ase.io.ulm abc.gpw

* Simple syntax for specifying BZ paths introduced:
  ``kpts={'path': 'GXK', 'npoints': 50}``.

* Calculations with ``fixdensity=True`` no longer update the Fermi level.

* The GPAW calculator object has a new
  :meth:`~ase.calculators.calculator.Calculator.band_structure`
  method that returns an :class:`ase.dft.band_structure.BandStructure`
  object.  This makes it very easy to create band-structure plots as shown
  in section 9 of this awesome Psi-k *Scientfic Highlight Of The Month*:
  http://psi-k.net/download/highlights/Highlight_134.pdf.

* Dipole-layer corrections for slab calculations can now be done in PW-mode
  also.  See :ref:`dipole`.

* New :meth:`~gpaw.paw.PAW.get_electrostatic_potential` method.

* When setting the default PAW-datasets or basis-sets using a dict, we
  must now use ``'default'`` as the key instead of ``None``:

  >>> calc = GPAW(basis={'default': 'dzp', 'H': 'sz(dzp)'})

  and not:

  >>> calc = GPAW(basis={None: 'dzp', 'H': 'sz(dzp)'})

  (will still work, but you will get a warning).


Version 1.1.0
=============

22 June 2016: :git:`1.1.0 <../1.1.0>`.

* Corresponding ASE release: ASE-3.11.0.

* There was a **BUG** in the recently added spin-orbit module.  Should now
  be fixed.

* The default Davidson eigensolver can now parallelize over bands.

* There is a new PAW-dataset file available:
  :ref:`gpaw-setup-0.9.20000.tar.gz <datasets>`.
  It's identical to the previous
  one except for one new data-file which is needed for doing vdW-DF
  calculations with Python 3.

* Jellium calculations can now be done in plane-wave mode and there is a new
  ``background_charge`` keyword (see the :ref:`Jellium tutorial <jellium>`).

* New band structure unfolding tool and :ref:`tutorial <unfolding tutorial>`.

* The :meth:`~gpaw.calculator.GPAW.get_pseudo_wave_function` method
  has a new keyword:  Use ``periodic=True`` to get the periodic part of the
  wave function.

* New tool for interpolating the pseudo wave functions to a fine real-space
  grids and for adding PAW-corrections in order to obtain all-electron wave
  functions.  See this tutorial: :ref:`ps2ae`.

* New and improved dataset pages (see :ref:`periodic table`).  Now shows
  convergence of absolute and relative energies with respect to plane-wave
  cut-off.

* :ref:`wannier90 interface`.

* Updated MacOSX installation guide for :ref:`homebrew` users.

* topological index


Version 1.0.0
=============

17 March 2016: :git:`1.0.0 <../1.0.0>`.

* Corresponding ASE release: ASE-3.10.0.

* A **BUG** related to use of time-reversal symmetry was found in the
  `G_0W_0` code that was introduced in version 0.11.  This has been `fixed
  now`_ --- *please run your calculations again*.

* New :mod:`gpaw.external` module.

* The gradients of the cavity and the dielectric in the continuum
  solvent model are now calculated analytically for the case of the
  effective potential method. This improves the accuracy of the forces
  in solution compared to the gradient calculated by finite
  differences. The solvation energies are expected to change slightly
  within the accuracy of the model.

* New `f_{\text{xc}}` kernels for correlation energy calculations.  See this
  updated :ref:`tutorial <rapbe_tut>`.

* Correlation energies within the range-separated RPA.  See this
  :ref:`tutorial <rangerpa_tut>`.

* Experimental interface to the libvdwxc_ library
  for efficient van der Waals density functionals.

* It's now possible to use Davidson and CG eigensolvers for MGGA calculations.

* The functional name "M06L" is now deprecated.  Use "M06-L" from now on.


.. _fixed now: https://gitlab.com/gpaw/gpaw/commit/c72e02cd789
.. _libvdwxc: https://gitlab.com/libvdwxc/libvdwxc


Version 0.11.0
==============

22 July 2015: :git:`0.11.0 <../0.11.0>`.

* Corresponding ASE release: ASE-3.9.1.

* When searching for basis sets, the setup name if any is now
  prepended automatically to the basis name.  Thus if
  :file:`setups='<setupname>'` and :file:`basis='<basisname>'`, GPAW
  will search for :file:`<symbol>.<setupname>.<basisname>.basis`.

* :ref:`Time-propagation TDDFT with LCAO <lcaotddft>`.

* Improved distribution and load balance when calculating atomic XC
  corrections, and in LCAO when calculating atomic corrections to the
  Hamiltonian and overlap.

* Norm-conserving :ref:`SG15 pseudopotentials <manual_setups>` and
  parser for several dialects of the UPF format.

* Non-selfconsistent spin-orbit coupling have been added. See :ref:`tutorial
  <spinorbit>` for examples of band structure calculations with spin-orbit
  coupling.

* Text output from ground-state calculations now list the symmetries found
  and the **k**-points used.  Eigenvalues and occupation numbers are now
  also printed for systems with **k**-points.

* :ref:`GW <gw exercise>`, :ref:`rpa`, and :ref:`response function
  calculation <df_tutorial>` has been rewritten to take advantage of
  symmetry and fast matrix-matrix multiplication (BLAS).

* New :ref:`symmetry <manual_symmetry>` keyword.  Replaces ``usesymm``.

* Use non-symmorphic symmetries: combining fractional translations with
  rotations, reflections and inversion.  Use
  ``symmetry={'symmorphic': False}`` to turn this feature on.

* New :ref:`forces <manual_convergence>` keyword in convergence.  Can
  be used to calculate forces to a given precision.

* Fixed bug in printing work functions for calculations with a
  dipole-correction `<http://listserv.fysik.dtu.dk/pipermail/
  gpaw-users/2015-February/003226.html>`_.

* A :ref:`continuum solvent model <continuum_solvent_model>` was added.

* A :ref:`orbital-free DFT <ofdft>` with PAW transformation is available.

* GPAW can now perform :ref:`electrodynamics` simulations using the
  quasistatic finite-difference time-domain (QSFDTD) method.

* BEEF-vdW, mBEEF and mBEEF-vdW functionals added.

* Support for Python 3.


Version 0.10.0
==============

8 April 2014: :git:`0.10.0 <../0.10.0>`.

* Corresponding ASE release: ASE-3.8.1

* Default eigensolver is now the Davidson solver.

* Default density mixer parameters have been changed for calculations
  with periodic boundary conditions.  Parameters for that case:
  ``Mixer(0.05, 5, 50)`` (or ``MixerSum(0.05, 5, 50)`` for spin-paired
  calculations.  Old parameters: ``0.1, 3, 50``.

* Default is now ``occupations=FermiDirac(0.1)`` if a
  calculation is periodic in at least one direction,
  and ``FermiDirac(0.0)`` otherwise (before it was 0.1 eV for anything
  with **k**-points, and 0 otherwise).

* Calculations with a plane-wave basis set are now officially supported.

* :ref:`One-shot GW calculations <gw_theory>` with full frequency
  integration or plasmon-pole approximation.

* Beyond RPA-correlation: `using renormalized LDA and PBE
  <https://trac.fysik.dtu.dk/projects/gpaw/browser/branches/sprint2013/doc/tutorials/fxc_correlation>`_.

* :ref:`bse theory`.

* Improved RMM-DIIS eigensolver.

* Support for new libxc 2.0.1.  libxc must now be built separately from GPAW.

* MGGA calculations can be done in plane-wave mode.

* Calculation of the stress tensor has been implemented for plane-wave
  based calculation (except MGGA).

* MGGA: number of neighbor grid points to use for FD stencil for
  wave function gradient changed from 1 to 3.

* New setups: Y, Sb, Xe, Hf, Re, Hg, Tl, Rn

* Non self-consistent calculations with screened hybrid functionals
  (HSE03 and HSE06) can be done in plane-wave mode.

* Modified setups:

  .. note::

     Most of the new semicore setups currently require
     :ref:`eigensolver <manual_eigensolver>` ``dav``, ``cg``
     eigensolvers or ``rmm-diis`` eigensolver with a couple of iterations.

  - improved eggbox: N, O, K, S, Ca, Sc, Zn, Sr, Zr, Cd, In, Sn, Pb, Bi

  - semicore states included: Na, Mg, V, Mn, Ni,
    Nb, Mo, Ru (seems to solve the Ru problem :git:`gpaw/test/big/Ru001/`),
    Rh, Pd, Ag, Ta, W, Os, Ir, Pt

  - semicore states removed: Te

  - elements removed: La (energetics was wrong: errors ~1eV per unit cell
    for PBE formation energy of La2O3 wrt. PBE benchmark results)

  .. note::

     For some of the setups one has now a choice of different
     number of valence electrons, e.g.::

       setups={'Ag': '11'}

     See :ref:`manual_setups` and list the contents of :envvar:`GPAW_SETUP_PATH`
     for available setups.

* new ``dzp`` basis set generated for all the new setups, see
  https://trac.fysik.dtu.dk/projects/gpaw/ticket/241


Version 0.9.0
=============

7 March 2012: :git:`0.9.0 <../0.9.0>`.

* Corresponding ASE release: ase-3.6

* Convergence criteria for eigenstates changed: The missing volume per
  grid-point factor is now included and the units are now eV**2. The
  new default value is 4.0e-8 eV**2 which is equivalent to the old
  default for a grid spacing of 0.2 Å.

* GPAW should now work also with NumPy 1.6.

* Much improved :ref:`command line tool` now based on the `new
  tool`_ in ASE.


.. _new tool: https://wiki.fysik.dtu.dk/ase/ase/cmdline.html


Version 0.8.0
=============

25 May 2011: :git:`0.8.0 <../0.8.0>`.

* Corresponding ASE release: ase-3.5.1
* Energy convergence criterion changed from 1 meV/atom to 0.5
  meV/electron.  This was changed in order to allow having no atoms like
  for jellium calculations.
* Linear :ref:`dielectric response <df_theory>` of an extended system
  (RPA and ALDA kernels) can now be calculated.
* :ref:`rpa`.
* Non-selfconsistent calculations with k-points for hybrid functionals.
* Methfessel-Paxton distribution added.
* Text output now shows the distance between planes of grid-points as
  this is what will be close to the grid-spacing parameter *h* also for
  non-orthorhombic cells.
* Exchange-correlation code restructured.  Naming convention for
  explicitely specifying libxc functionals has changed: :ref:`manual_xc`.
* New PAW setups for Rb, Ti, Ba, La, Sr, K, Sc, Ca, Zr and Cs.


Version 0.7.2
=============

13 August 2010: :git:`0.7.2 <../0.7.2>`.

* Corresponding ASE release: ase-3.4.1
* For version 0.7, the default Poisson solver was changed to
  ``PoissonSolver(nn=3)``.  Now, also the Poisson solver's default
  value for ``nn`` has been changed from ``'M'`` to ``3``.


Version 0.7
===========

23 April 2010: :git:`0.7 <../0.7>`.

* Corresponding ASE release: ase-3.4.0
* Better and much more efficient handling of non-orthorhombic unit
  cells.  It may actually work now!
* Much better use of ScaLAPACK and BLACS.  All large matrices can now
  be distributed.
* New test coverage pages for all files.
* New default value for Poisson solver stencil: ``PoissonSolver(nn=3)``.
* Much improved MPI module (:ref:`communicators`).
* Self-consistent Meta GGA.
* New :ref:`PAW setup tar-file <setups>` now contains revPBE setups and
  also dzp basis functions.
* New ``$HOME/.gpaw/rc.py`` configuration file.
* License is now GPLv3+.
* New HDF IO-format.
* :ref:`Advanced GPAW Test System <big-test>` Introduced.


Version 0.6
===========

9 October 2009: :git:`0.6 <../0.6>`.

* Corresponding ASE release: ase-3.2.0
* Much improved default parameters.
* Using higher order finite-difference stencil for kinetic energy.
* Many many other improvements like: better parallelization, fewer bugs and
  smaller memory footprint.


Version 0.5
===========

1 April 2009: :git:`0.5 <../0.5>`.

* Corresponding ASE release: ase-3.1.0
* `new setups added Bi, Br, I, In, Os, Sc, Te; changed Rb setup <https://trac.fysik.dtu.dk/projects/gpaw/changeset/3612>`_.
* `memory estimate feature is back <https://trac.fysik.dtu.dk/projects/gpaw/changeset/3575>`_


Version 0.4
===========

13 November 2008: :git:`0.4 <../0.4>`.

* Corresponding ASE release: ase-3.0.0
* Now using ASE-3 and numpy.
* TPSS non self-consistent implementation.
* LCAO mode.
* VdW-functional now coded in C.
* Added atomic orbital basis generation scripts.
* Added an Overlap object, and moved apply_overlap and apply_hamiltonian
  from Kpoint to Overlap and Hamiltonian classes.

* Wannier code much improved.
* Experimental LDA+U code added.
* Now using libxc.
* Many more setups.
* Delta scf calculations.

* Using localized functions will now no longer use MPI group
  communicators and blocking calls to MPI_Reduce and MPI_Bcast.
  Instead non-blocking sends/receives/waits are used.  This will
  reduce synchronization time for large parallel calculations.

* More work on LB94.
* Using LCAO code forinitial guess for grid calculations.
* TDDFT.
* Moved documentation to Sphinx.
* Improved metric for Pulay mixing.
* Porting and optimization for BlueGene/P.
* Experimental Hartwigsen-Goedecker-Hutter pseudopotentials added.
* Transport calculations with LCAO.


Version 0.3
===========

19 December 2007: :git:`0.3 <../0.3>`.
