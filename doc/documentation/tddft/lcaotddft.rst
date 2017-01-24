.. _lcaotddft:

===================================================
Time-propagation TDDFT with LCAO : Theory and usage
===================================================

This page documents the use of time-propagation TDDFT in :ref:`LCAO
mode <lcao>`. The implementation is described in Ref. [#Kuisma2015]_.

Real time propagation of LCAO-functions
=======================================

In the real-time LCAO-TDDFT approach, the time-dependent wave functions are
represented using the
localized basis functions `\tilde{\phi}_{\mu}(\mathbf r)` as

.. math::

  \tilde{\psi}_n(\mathbf{r},t) = \sum_{\mu} \tilde{\phi}_{\mu}(\mathbf{r}-\mathbf{R}^\mu) c_{\mu n}(t) .

The time-dependent Kohn--Sham equation in the PAW formalism can be written as

.. math::

  \left[ \widehat T^\dagger \left( -i \frac{{\rm d}}{{\rm d}t} + \hat H_{\rm KS}(t) \right) \widehat T \right]  \tilde{\Psi(\mathbf{r},t)} = 0.

From this, the following matrix equation can be derived for the
LCAO wave function coefficients:

.. math::
  {\rm i}\mathbf{S} \frac{{\rm d}\mathbf{C}(t)}{{\rm d}t} = \mathbf{H}(t) \mathbf{C}(t).

In the current implementation, `\mathbf C`, `\mathbf S` and
`\mathbf H` are dense matrices which are distributed using
ScaLAPACK/BLACS.  Currently, a semi-implicit Crank--Nicolson method (SICN) is
used to propagate the wave functions. For wave functions at time `t`, one
propagates the system forward using `\mathbf H(t)` and solving the
linear equation

.. math::

  \left( \mathbf{S} + {\rm i} \mathbf{H}(t) {\rm d}t / 2 \right) \mathbf{C}'(t+{\rm d}t) = \left( \mathbf{S} - {\rm i} \mathbf{H}(t) {\rm d}t / 2 \right) \mathbf{C}(t).

Using the predicted wave functions `C'(t+\mathrm dt)`, the
Hamiltonian `H'(t+\mathrm dt)` is calculated and the Hamiltonian at
middle of the time step is estimated as

.. math::

   \mathbf{H}(t+{\rm d}t/2) = (\mathbf{H}(t) + \mathbf{H}'(t+{\rm d}t)) / 2

With the improved Hamiltonian, the wave functions are again propagated
from `t` to `t+\mathrm dt` by solving

.. math::

  \left( \mathbf{S} + {\rm i} \mathbf{H}(t+{\rm d}t/2) {\rm d}t / 2 \right) \mathbf{C}(t+{\rm d}t) = \left( \mathbf{S} - {\rm i} \mathbf{H}(t+{\rm d}t/2) {\rm d}t / 2 \right) \mathbf{C}(t).

This procedure is repeated using 500--2000 time steps of 5--40 as to
calculate the time evolution of the electrons.

Usage
=====

Create an LCAOTDDFT object like a GPAW calculator::

 >>> from gpaw.lcaotddft import LCAOTDDFT
 >>> td_calc = LCAOTDDFT(setups={'Na': '1'}, basis='dzp', xc='LDA', h=0.3,
                         nbands=1, convergence={'density': 1e-7},
                         poissonsolver=PoissonSolver(eps=1e-20,
                                                     remove_moment=1 + 3 + 5))

Some important points are:

 * The grid spacing is only used to calculate the Hamiltonian matrix and therefore a coarser grid than usual can be used.
 * Completely unoccupied bands should be left out of the calculation, since they are not needed.
 * The density convergence criterion should be a few orders of magnitude more accurate than in ground state calculations.
 * The convergence tolerance of the Poisson solver should be at least 1e-14, but 1e-20 does not hurt (note that this is the *quadratic* error).
 * One should use multipole-corrected Poisson solvers in any TDDFT run. See the documentation on :ref:`advancedpoisson`.

Perform a regular ground state calculation and get the ground state
wave functions::

 >>> atoms.set_calculator(td_calc)
 >>> atoms.get_potential_energy()

If you wish to save here, write the wave functions also::

 >>> td_calc.write('Na2.gpw', mode='all')

The calculation proceeds as in grid mode. We kick the system in the x
direction and propagate 500 steps of 10 as::

 >>> td_calc.absorption_kick([1e-5, 0.0, 0.0])
 >>> td_calc.propagate(10, 500, out='Na2.dm')

The spectrum is obtained in the same manner as in FD time-propagations.

Simple example script
=====================

.. literalinclude:: lcaotddft.py


General notes about basis sets
==============================

In time-propagation LCAO-TDDFT, it is much more important to think
about the basis sets compared to ground state LCAO calculations.  It
is required that the basis set can represent both the occupied
(holes) and relevant unoccupied states (electrons) adequately.  Custom
basis sets for the time propagation should be generated according to
one's need, and then benchmarked.

**Irrespective of the basis sets you choose, ALWAYS, ALWAYS, benchmark LCAO
results with respect to the FD time-propagation code** on the largest system
possible. For example, one can create a prototype system which consists of
similar atomic species with similar roles as in the parent system, but small
enough to calculate with grid propagation mode.
See Figs. 4 and 5 of Ref. [#Kuisma2015]_ for example benchmarks.

After these remarks, we describe two packages of basis sets that can be used as a
starting point for choosing a suitable basis set for your needs. Namely,
:ref:`pvalence basis sets` and :ref:`coopt basis sets`.


.. _pvalence basis sets:

p-valence basis sets
--------------------

The so-called p-valence basis sets are constructed for transition metals by
replacing the p-type polarization function of the default basis sets with a
bound unoccupied p-type orbital and its split-valence complement. Such basis
sets correspond to the ones used in Ref. [#Kuisma2015]_. These basis sets
significantly improve density of states of unoccupied states.

The p-valence basis sets can be easily obtained for the appropriate elements
with the :command:`gpaw install-data` tool using the following options::

    $ gpaw install-data {<directory>} --basis --version=pvalence
    
See :ref:`installation of paw datasets` for more information on basis set
installation. It is again reminded that these basis sets are not thoroughly
tested and **it is essential to benchmark the performance of the basis sets
for your application**.


.. _coopt basis sets:

Completeness-optimized basis sets
---------------------------------

A systematic approach for improving the basis sets can be obtained with the
so-called completeness-optimization approach. This approach is used in Ref.
[#Rossi2015]_ to generate basis set series for TDDFT calculations of copper,
silver, and gold clusters.

For further details of the basis sets, as well as their construction and
performance, see [#Rossi2015]_. For convenience, these basis sets can be easily
obtained with::
    
    $ gpaw install-data {<directory>} --basis --version=coopt
    
See :ref:`installation of paw datasets` for basis set installation. Finally,
it is again emphasized that when using the basis sets, **it is essential to
benchmark their suitability for your application**.


Parallelization
===============

LCAO-TDDFT is parallelized using ScaLAPACK. It runs without ScaLAPACK, but in this case only a single core is used for linear alrebra.

 * Use ``parallel={'sl_default':(N,M,64)}``;  See :ref:`manual_parallel`.
 * ScaLAPACK can be also enabled by specifying --sl_default=N,M,64 in command line.
 * It is necessary that N*M equals the total number of cores used by the calculator, and ``max(N,M)*64 < nbands``. The block size 64 can be changed to, e.g., 16 if necessary.
 * Apart from parallelization of linear algrebra, normal domain and band parallelizations can be used. As in ground-state LCAO calculations, use band parallelization to reduce memory consumption.

Timing
======

TODO: add ``ParallelTimer`` example


Advanced tutorial - Plasmon resonance of silver cluster
=======================================================

One should think about what type of transitions of interest are present, and make sure
that the basis set can represent such Kohn-Sham electron and hole wave functions.
The first transitions in silver clusters will be `5s \rightarrow 5p` like. We
require 5p orbitals in the basis set, and thus, we must generate a custom basis
set.

Here is how to generate a double-zeta basis set with 5p orbital in valence for
Silver for GLLB-SC potential. Note that the extra 5p valence state effectively improves on the ordinary polarization function, so this basis set is *better* than the default double-zeta polarized one.  We will use the 10-electron Ag setup, since the semi-core p states included in the default setup are not relevant here.

.. literalinclude:: lcaotddft_basis.py

We calculate the icosahedral Ag55 cluster: :download:`ag55.xyz`

This code uses ScaLAPACK parallelization with 64 cores.

.. literalinclude:: lcaotddft_ag55.py

Code runs for approximately two hours wall-clock time. The resulting spectrum shows already emerging plasmonic excitation around 4 eV.
For more details, see [#Kuisma2015]_.

.. image:: fig1.png

Induced density
===============

Plotting the induced density is especially interesting in case of plasmon resonances. As an example, we calculate a dummy Na8 wire and write the density to a file on every iteration.
There are certain advantages of writing the density on every iteration instead of using the predefined frequencies and on-the-fly Fourier transformation: Only one TDDFT run is required as any frequency can be analysed as a post processing operation.
Hard disk requirements are large, but tolerable (1-100GB) in most cases.

.. literalinclude:: lcaotddft_induced.py

Files with extensions ``.sG`` and ``.asp`` are created, where ``.sG`` files contain the density on the coarse grid while ``.asp`` files contain the atomic density matrices. With these, it is possible to reconstruct the full density.
This can now be fourier transformed at the desired frequency. Here, we look from the produced spectrum file that plasmonic peak, and perform Fourier transform at that frequency.

.. literalinclude:: lcaotddft_analyse.py

Two cube files are created, one for the sin (imag) and cos (real) transform at the frequency.  Usually, one is interested in the absorbing part, i.e., the imaginary part. Below the plasmon resonance is visualized
in the Na8 wire. In their current form, these cube files contain just the pseudo part of density.

.. image:: Na8_imag.png


Advanced tutorial - large organic molecule
==========================================

General notes
-------------

On large organic molecules, on large conjugated systems, there will `\pi \rightarrow \pi^*`,
`\sigma \rightarrow \sigma^*`. These states consist of only
the valence orbitals of carbon, and they are likely by quite similar few eV's
below and above the fermi lavel. These are thus a reason to believe that these
states are well described with hydrogen 1s and carbon 2s and 2p valence orbitals
around the fermi level.

Here, we will calculate a small and a large organic molecule with lcao-tddft.

**TODO**

Kohn-Sham decomposition of the transition density matrix
========================================================

Soon it will be possible to analyse the origin of the transitions the same way as is commonly done in Casida-based codes.
The LCAO basis will be transformed to an electron-hole basis of the Kohn-Sham system.


References
==========

.. [#Kuisma2015]
   M. Kuisma, A. Sakko, T. P. Rossi, A. H. Larsen, J. Enkovaara, L. Lehtovaara, and T. T. Rantala,
   Localized surface plasmon resonance in silver nanoparticles: Atomistic first-principles time-dependent
   density functional theory calculations,
   *Phys. Rev. B* **69**, 245419 (2004).
   `doi:10.1103/PhysRevB.91.115431 <http://dx.doi.org/10.1103/PhysRevB.91.115431>`_

.. [#Rossi2015]
   T. P. Rossi, S. Lehtola, A. Sakko, M. J. Puska, and R. M. Nieminen,
   Nanoplasmonics simulations at the basis set limit through completeness-optimized, local numerical basis sets,
   *J. Chem. Phys.* **142**, 094114 (2015).
   `doi:10.1063/1.4913739 <http://dx.doi.org/10.1063/1.4913739>`_
