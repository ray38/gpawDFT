from ase.io import read
from gpaw.tddft import photoabsorption_spectrum
from gpaw import PoissonSolver
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.occupations import FermiDirac
from gpaw import setup_paths
setup_paths.insert(0, '.')

atoms = read('ag55.xyz')
atoms.center(vacuum=5.0)

# Increase accuragy of density for ground state
convergence = {'density': 1e-6}

# Increase accuracy of Poisson Solver and apply multipole corrections up to l=2
poissonsolver = PoissonSolver(eps=1e-14, remove_moment=1 + 3)

td_calc = LCAOTDDFT(xc='GLLBSC', basis='GLLBSC.dz', h=0.3, nbands=352,
                    convergence=convergence, poissonsolver=poissonsolver,
                    occupations=FermiDirac(0.1),
                    parallel={'sl_default': (8, 8, 32), 'band': 2})

atoms.set_calculator(td_calc)
# Relax the ground state
atoms.get_potential_energy()

# For demonstration purposes, save intermediate ground state result to a file
td_calc.write('ag55.gpw', mode='all')

td_calc = LCAOTDDFT('ag55.gpw',
                    parallel={'sl_default': (8, 8, 32), 'band': 2})

td_calc.absorption_kick([1e-5, 0.0, 0.0])
td_calc.propagate(20, 500, 'ag55.dm')

photoabsorption_spectrum('ag55.dm', 'ag55.spec', width=0.2)
