.. _optimizer_tests:

===============
Optimizer tests
===============
This page shows benchmarks of optimizations done with our different optimizers.
Note that the iteration number (steps) is not the same as the number of force
evaluations. This is because some of the optimizers uses internal line searches
or similar.

The most important performance characteristics of an optimizer is the
total optimization time.
Different optimizers may perform the same number of steps, but along a different
path, so the time spent on calculation of energy/forces may be different
due to different convergence of the self-consistent field.

G2
==
PBE relaxation of molecules from the G2 set.

On the plots: the number of optimizer force calls (stats), the total run time,
the systems with the largest number of optimizer force calls, and the number of
systems for which optimization failed.

In the corresponding tables: the most common value of the total energy
("relaxed energy"), and the differences (optimizer - "relaxed energy").
The dot sign denotes the above difference below a threshold
(same as the printed precision of "relaxed energy" in eV),
and "N/A" denotes an optimization failure.
Only the systems that fail to converge or converge to a
total energy above the threshold are given in the tables.

Calculator used: GPAW mode='lcao' (see :git:`~doc/devel/ase_optimize/g2_dzp.py`)

Limit of optimizer steps: 25

.. csv-table::
   :file: g2_dzp.csv
   :header: optimizer, failed, time, steps


N2Cu
====
Relaxation of Cu surface.

Calculator used: EMT

.. csv-table::
   :file: N2Cu-surf.csv

N2 adsorption on relaxed Ru surface

Calculator used: EMT

.. csv-table::
   :file: N2Cu-N2.csv

Cu_bulk
=======
Bulk relaxation of Cu where atoms has been rattled.

Calculator used: EMT

.. csv-table::
   :file: Cu_bulk.csv

CO_Au111
========
CO adsorption on Au

Calculator used: EMT

.. csv-table::
   :file: CO_Au111.csv

H2
==
Geometry optimization of gas-phase molecule.

Calculator used: EMT

.. csv-table::
   :file: H2-emt.csv

Calculator used: GPAW

.. csv-table::
   :file: H2-gpaw.csv

C5H12
=====
Geometry optimization of gas-phase molecule.

Calculator used: GPAW (lcao)

.. csv-table::
   :file: C5H12-gpaw.csv

nanoparticle
============
Adsorption of a NH on a Pd nanoparticle.

Calculator used: GPAW (lcao)

.. csv-table::
   :file: nanoparticle.csv

NEB
=======
Diffusion of gold atom on Al(100) surface.

Calculator used: EMT

.. csv-table::
   :file: neb-emt.csv

Calculator used: GPAW (lcao)

.. csv-table::
   :file: neb-gpaw.csv
