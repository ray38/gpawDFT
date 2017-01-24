"""Band structure tutorial

Calculate the band structure of Si along high symmetry directions
Brillouin zone
"""

from ase.build import bulk
from gpaw import GPAW, PW, FermiDirac

# Perform standard ground state calculation (with plane wave basis)
si = bulk('Si', 'diamond', 5.43)
calc = GPAW(mode=PW(200),
            xc='PBE',
            kpts=(8, 8, 8),
            random=True,  # random guess (needed if many empty bands required)
            occupations=FermiDirac(0.01),
            txt='Si_gs.txt')
si.calc = calc
si.get_potential_energy()
calc.write('Si_gs.gpw')

# Restart from ground state and fix potential:
calc = GPAW('Si_gs.gpw',
            nbands=16,
            fixdensity=True,
            symmetry='off',
            kpts={'path': 'GXWKL', 'npoints': 60},
            convergence={'bands': 8})

calc.get_potential_energy()

band_structure = calc.band_structure()
band_structure.plot(filename='bandstructure.png',
                    show=True)
