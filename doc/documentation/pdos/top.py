from ase.build import fcc111, add_adsorbate
from gpaw import GPAW, PW


#  Slab with CO:
slab = fcc111('Pt', size=(1, 1, 3))
add_adsorbate(slab, 'C', 2.0, 'ontop')
add_adsorbate(slab, 'O', 3.15, 'ontop')
slab.center(axis=2, vacuum=4.0)
slab.calc = GPAW(mode=PW(400),
                 xc='RPBE',
                 kpts=(12, 12, 1),
                 #eigensolver='cg',
                 #mixer=Mixer(0.05, 5, 100),
                 convergence={#'energy': 100,
                 #             'density': 100,
                 #             'eigenstates': 1.0e-7,
                             'bands': -10},
                 txt='top.txt')
slab.get_potential_energy()
slab.calc.write('top.gpw', mode='all')

#  Molecule
molecule = slab[:-2]
molecule.calc = GPAW(mode=PW(400),
                     xc='RPBE',
                     kpts=(12, 12, 1),
                     #eigensolver='cg',
                     #convergence={'energy': 100,
                     #             'density': 100,
                     #             'eigenstates': 1.0e-9,
                     #             'bands': 'occupied'},
                     txt='CO.txt')

molecule.get_potential_energy()
molecule.calc.write('CO.gpw', mode='all')
