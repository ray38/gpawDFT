.. _Niflheim:

========
Niflheim
========

Information about the Niflheim cluster can be found at
`<https://wiki.fysik.dtu.dk/niflheim>`_.

.. highlight:: bash

Installing GPAW on all three architectures
==========================================

Here, we install the development versions of ASE and GPAW in ``~/ase`` and
``~/gpaw``.  First, make sure you have this in your ``~/.bashrc``:
    
.. literalinclude:: gpaw.sh
    
Now, get the source code for ASE and GPAW::

    $ cd
    $ rm -rf gpaw ase
    $ git clone https://gitlab.com/ase/ase.git
    $ git clone https://gitlab.com/gpaw/gpaw.git
    
and compile GPAW's C-extension using the :download:`compile.sh` script::
    
    $ cd gpaw
    $ sh doc/platforms/Linux/Niflheim/compile.sh

Submit jobs to the queue with::
    
    $ gpaw-qsub -c 16 -q long my-script.py
    
Type ``gpaw-qsub -h`` for help.  Note that ``gpaw-qsub`` is set up as an alias
for this script :download:`gpaw-qsub.py`.


Using more than one version of GPAW
===================================

Here we install an additional version of GPAW for, say, production runs::
    
    $ cd
    $ mkdir production
    $ cd production
    $ git clone https://gitlab.com/gpaw/gpaw.git
    $ cd gpaw
    $ git checkout 1.0.1
    $ sh doc/platforms/Linux/Niflheim/compile.sh
    
Now you can submit jobs that use this production version with::

    $ gpaw-qsub -c 16 -q long -g ~/production/gpaw my-script.py
