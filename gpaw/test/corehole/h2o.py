from __future__ import print_function
import numpy as np
from math import pi, cos, sin
from ase import Atom, Atoms
# from ase.parallel import rank, barrier
from gpaw import GPAW, PoissonSolver
from gpaw.xas import XAS
from gpaw.test import equal, gen

# Generate setup for oxygen with half a core-hole:
gen('O', name='hch1s', corehole=(1, 0, 0.5))

a = 5.0
d = 0.9575
t = pi / 180 * 104.51
H2O = Atoms([Atom('O', (0, 0, 0)),
             Atom('H', (d, 0, 0)),
             Atom('H', (d * cos(t), d * sin(t), 0))],
            cell=(a, a, a), pbc=False)
H2O.center()
calc = GPAW(nbands=10, h=0.2, setups={'O': 'hch1s'},
            poissonsolver=PoissonSolver(use_charge_center=True))
H2O.set_calculator(calc)
e = H2O.get_potential_energy()
niter = calc.get_number_of_iterations()

import gpaw.mpi as mpi

if mpi.size == 1:
    xas = XAS(calc)
    x, y = xas.get_spectra()
    e1_n = xas.eps_n
    de1 = e1_n[1] - e1_n[0]

calc.write('h2o-xas.gpw')

if mpi.size == 1:
    calc = GPAW('h2o-xas.gpw', txt=None,
                poissonsolver=PoissonSolver(use_charge_center=True))
    calc.initialize()
    xas = XAS(calc)
    x, y = xas.get_spectra()
    e2_n = xas.eps_n
    w_n = np.sum(xas.sigma_cn.real**2, axis=0)
    de2 = e2_n[1] - e2_n[0]

    print(de2)
    print(de2 - 2.0705)
    assert abs(de2 - 2.0705) < 0.001
    print(w_n[1] / w_n[0])
    assert abs(w_n[1] / w_n[0] - 2.22) < 0.01

    if mpi.size == 1:
        assert de1 == de2


if 0:
    import pylab as p
    p.plot(x, y[0])
    p.plot(x, sum(y))
    p.show()

print(e, niter)
energy_tolerance = 0.00009
niter_tolerance = 0
equal(e, -17.9772, energy_tolerance)
