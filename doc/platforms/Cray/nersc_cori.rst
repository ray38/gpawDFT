.. _nersc_cori:

=====================
cori.nersc.gov (XC40)
=====================


.. note::
   These instructions are up-to-date as of January 2017.

GPAW
====

At NERSC it is recommened to install GPAW on Cori with Anaconda python. For
massivly parallel applications it is recommened to use `Shifter
<http://www.nersc.gov/research-and-development/user-defined-images/>`_.

GPAW can be built with a minimal ``customize.py``

.. literalinclude:: customize_nersc_cori.py


Load the GNU programming environment and set Cray environment for dynamic
linking::

  export CRAYPE_LINK_TYPE=dynamic
  module rm PrgEnv-intel
  module load PrgEnv-gnu
  module load cray-hdf5-parallel
  module load python/2.7-anaconda


Install ASE with pip while the Anaconda python module is loaded::

  pip install ase --user

Build and install GPAW::

  python setup.py build_ext
  python setup.py install --user


To setup the environment::

  export PATH=$HOME/.local/cori/2.7-anaconda/bin:$PATH
  export OMP_NUM_THREADS=1
  export PYTHONHOME=/global/common/cori/software/python/2.7-anaconda


Then the test suite can be run from a batch script or interactive session with::

  srun -n 8 -c 2 --cpu_bind=cores gpaw-python `which gpaw` test
