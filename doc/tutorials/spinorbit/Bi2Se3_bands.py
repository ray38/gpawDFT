from __future__ import print_function
from ase.dft.kpoints import bandpath
from ase.io import read
from ase.parallel import paropen
from gpaw import GPAW

a = read('gs_Bi2Se3.gpw')

G = [0.0, 0.0, 0.0]
L = [0.5, 0.0, 0.0]
F = [0.5, 0.5, 0.0]
Z = [0.5, 0.5, 0.5]
kpts, x, X = bandpath([G, Z, F, G, L], a.cell, npoints=200)

calc = GPAW('gs_Bi2Se3.gpw',
            kpts=kpts,
            symmetry='off',
            parallel={'band': 16},
            txt='Bi2Se3_bands.txt')
calc.diagonalize_full_hamiltonian(nbands=48, scalapack=(4, 4, 32))

calc.write('Bi2Se3_bands.gpw', mode='all')

with paropen('kpath.dat', 'w') as f:
    for k in x:
        print(k, file=f)

with paropen('highsym.dat', 'w') as f:
    for k in X:
        print(k, file=f)
