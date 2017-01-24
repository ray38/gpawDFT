from gpaw import GPAW

calc = GPAW('Pt_gs.gpw', txt=None)

calc = GPAW('Pt_gs.gpw',
            kpts={'path': 'GXWLGKX', 'npoints': 200},
            symmetry='off',
            txt='Pt_bands.txt')
calc.diagonalize_full_hamiltonian(nbands=20)
calc.write('Pt_bands.gpw', mode='all')
