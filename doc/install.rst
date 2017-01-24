.. _installation:

============
Installation
============

.. toctree::
    :hidden:

    troubleshooting
    platforms/platforms

GPAW relies on the Python library *atomic simulation environment* (ASE_),
so you need to :ref:`install ASE <ase:download_and_install>` first.  GPAW
itself is written mostly in the Python programming language, but there
are also some C-code used for:

* performance critical parts
* allowing Python to talk to external numerical libraries (BLAS_, LAPACK_,
  LibXC_, MPI_ and ScaLAPACK_)

So, in order to make GPAW work, you need to compile some C-code.  For serial
calculations, you will need to build a dynamically linked library
(``_gpaw.so``) that the standard Python interpreter can load.  For parallel
calculations, you need to build a new Python interpreter (``gpaw-python``)
that has MPI_ functionality built in.

There are several ways to install GPAW:

* On a lucky day it's as simple as ``pip install -U gpaw`` as
  described :ref:`below <installation using pip>`.

* Alternatively, you can :ref:`download <download>` the source code,
  edit :git:`customize.py` to tell the install script which libraries you
  want to link to and where they
  can be found (see :ref:`customizing installation`) and then install with a
  ``python setup.py install --user`` as described :ref:`here <install
  with distutils>`.

* There may be a package for your Linux distribution that you can use
  (named ``gpaw``).

* If you are a developer that need to change the code you should look at this
  description: :ref:`developer installation`.

.. seealso::

    * Using :ref:`homebrew` on MacOSX.
    * Tips and tricks for installation on many :ref:`platforms and
      architectures`.
    * :ref:`troubleshooting`.
    * Important :ref:`envvars`.
    * In case of trouble: :ref:`Our mail list and IRC channel <contact>`.


Requirements
============

* Python_ 2.6-3.5
* NumPy_ 1.6.1 or later (base N-dimensional array package)
* ASE_ 3.11 or later (atomic simulation environment)
* a C-compiler
* LibXC_ 2.0.1 or later
* BLAS_ and LAPACK_ libraries

Optional, but highly recommended:

* SciPy_ 0.7 or later (library for scientific computing, requirered for
  some features)
* an MPI_ library (required for parallel calculations)
* FFTW_ (for increased performance)
* BLACS_ and ScaLAPACK_


.. _Python: http://www.python.org/
.. _NumPy: http://docs.scipy.org/doc/numpy/reference/
.. _SciPy: http://docs.scipy.org/doc/scipy/reference/
.. _LibXC: http://www.tddft.org/programs/octopus/wiki/index.php/Libxc
.. _MPI: http://www.mpi-forum.org/
.. _BLAS: http://www.netlib.org/blas/
.. _BLACS: http://www.netlib.org/blacs/
.. _LAPACK: http://www.netlib.org/lapack/
.. _ScaLAPACK: http://www.netlib.org/scalapack/
.. _PyPI: https://pypi.python.org/pypi/gpaw
.. _PIP: https://pip.pypa.io/en/stable/
.. _ASE: https://wiki.fysik.dtu.dk/ase
.. _FFTW: http://www.fftw.org/


.. _installation using pip:

Installation using ``pip``
==========================

.. highlight:: bash

The simplest way to install GPAW is using pip_ and the GPAW package from
the Python package index (PyPI_)::

    $ pip install --upgrade --user gpaw

This will compile and install GPAW (both ``_gpaw.so`` and all the Python
files) in your ``~/.local/lib/pythonX.Y/site-packages`` folder where
Python can automatically find it.  The ``pip`` command will also place
the command line tool :command:`gpaw` in the ``~/.local/bin`` folder, so
make sure you have that in your :envvar:`PATH` environment variable.  If
you have an ``mpicc`` command on your system then there will also be a
``gpaw-python`` executable in ``~/.local/bin``.

Check that you have installed everything in the correct places::

    $ gpaw info


Install PAW datasets
====================

Install the datasets into the folder ``<dir>`` using this command::

    $ gpaw install-data <dir>

See :ref:`installation of paw datasets` for more details.

Now you should be ready to use GPAW, but before you start, please run the
tests as described below.


.. index:: test
.. _run the tests:

Run the tests
=============

Make sure that everything works by running the test suite::

    $ gpaw test

This will take a couple of hours.  You can speed it up by using more than
one core::

    $ gpaw test -j 4

Please report errors to the ``gpaw-developers`` mailing list so that we
can fix them (see :ref:`mail lists`).

If tests pass, and the parallel version is built, test the parallel code::

    $ gpaw -P 4 test

or equivalently::

    $ mpiexec -np 4 gpaw-python `which gpaw` test


.. _download:

Getting the source code
=======================

Sou can get the source from a tar-file or from Git:

:Tar-file:

    You can get the source as a tar-file for the
    latest stable release (gpaw-1.1.0.tar.gz_) or the latest
    development snapshot (`<snapshot.tar.gz>`_).

    Unpack and make a soft link::

        $ tar -xf gpaw-1.1.0.tar.gz
        $ ln -s gpaw-1.1.0 gpaw

:Git clone:

    Alternatively, you can get the source for the the development version
    from https://gitlab.com/gpaw/gpaw like this::

        $ git clone https://gitlab.com/gpaw/gpaw.git

    If you want the latest stable release you should clone and then *checkout*
    the ``1.1.0`` tag like this::

        $ git clone https://gitlab.com/gpaw/gpaw.git
        $ git checkout 1.1.0

Add ``~/gpaw`` to your :envvar:`PYTHONPATH` environment variable and add
``~/gpaw/tools`` to :envvar:`PATH` (assuming ``~/gpaw`` is where your GPAW
folder is).

.. note::

    We also have Git tags for older stable versions of GPAW.
    See the :ref:`releasenotes` for which tags are available.  Also the
    dates of older releases can be found there.

.. _gpaw-1.1.0.tar.gz:
    https://pypi.python.org/packages/71/e6/
    d26db47ec7bc44d21fbefedb61a8572276358b50862da3390c20664d9511/
    gpaw-1.1.0.tar.gz


.. _customizing installation:

Customizing installation
========================

The install script does its best when trying to guess proper libraries
and commands to build GPAW. However, if the standard procedure fails
or user wants to override default values it is possible to customize
the setup with :git:`customize.py` file which is located in the GPAW base
directory. As an example, :git:`customize.py` might contain the following
lines::

  libraries = ['myblas', 'mylapack']
  library_dirs = ['path_to_myblas']

Now, GPAW would be built with "``-Lpath_to_myblas -lmyblas
-lmylapack``" linker flags. Look at the file :git:`customize.py`
itself for more possible options.  :ref:`platforms and architectures`
provides examples of :file:`customize.py` for different platforms.
After editing :git:`customize.py`, follow the instructions for the
:ref:`developer installation`.


.. _install with distutils:

Install with setup.py
=====================

If you have the source code, you can use the install script (:git:`setup.py`)
to compile and install the code::

    $ python setup.py install --user


.. _parallel installation:

Parallel installation
=====================

By default, setup looks if :program:`mpicc` is available, and if setup
finds one, a parallel version is build. If the setup does not find
mpicc, a user can specify one in the :git:`customize.py` file.

Additionally a user may want to enable ScaLAPACK, setting in
:git:`customize.py`::

    scalapack = True

and, in this case, provide BLACS/ScaLAPACK ``libraries`` and ``library_dirs``
as described in :ref:`customizing installation`.

Instructions for running parallel calculations can be found in the
:ref:`user manual <manual_parallel_calculations>`.


Libxc Installation
------------------

If you OS does not have a LibXC_ package you can use then you can download
and install LibXC_ as described `here
<http://www.tddft.org/programs/octopus/wiki/index.php/Libxc:download>`_.  A
few extra tips:

* Libxc installation requires both a C compiler and a fortran compiler.

* We've tried intel and gnu compilers and haven't noticed much of a
  performance difference.  Use whatever is easiest.

* Libxc shared libraries can be built with the "--enable-shared" option
  to configure.  This might be slightly preferred because it reduces
  memory footprints for executables.

* Typically when building GPAW one has to modify customize.py in a manner
  similar to the following::

    library_dirs += ['/my/path/to/libxc/2.0.2/install/lib']
    include_dirs += ['/my/path/to/libxc/2.0.2/install/include']

  or if you don't want to modify your customize.py, you can add these lines to
  your .bashrc::

    export C_INCLUDE_PATH=/my/path/to/libxc/2.0.2/install/include
    export LIBRARY_PATH=/my/path/to/libxc/2.0.2/install/lib
    export LD_LIBRARY_PATH=/my/path/to/libxc/2.0.2/install/lib

Example::

    wget http://www.tddft.org/programs/octopus/down.php?file=libxc/libxc-2.0.2.tar.gz -O libxc-2.0.2.tar.gz
    tar -xf libxc-2.0.2.tar.gz
    cd libxc-2.0.2
    ./configure --enable-shared --prefix=$HOME/xc
    make
    make install

    # add these to your .bashrc:
    export C_INCLUDE_PATH=~/xc/include
    export LIBRARY_PATH=~/xc/lib
    export LD_LIBRARY_PATH=~/xc/lib


.. _envvars:

Environment variables
=====================

.. envvar:: PATH

    Colon-separated paths where programs can be found.

.. envvar:: PYTHONPATH

    Colon-separated paths where Python modules can be found.

.. envvar:: OMP_NUM_THREADS

    Currently should be set to 1.

.. envvar:: GPAW_SETUP_PATH

    Comma-separated paths to folders containing the PAW datasets.

Set these permanently in your :file:`~/.bashrc` file::

    $ export PYTHONPATH=~/gpaw:$PYTHONPATH
    $ export PATH=~gpaw/tools:$PATH

or your :file:`~/.cshrc` file::

    $ setenv PYTHONPATH ${HOME}/gpaw:${PYTHONPATH}
    $ setenv PATH ${HOME}/gpaw/tools:${PATH}
