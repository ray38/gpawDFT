from gpaw.tddft import photoabsorption_spectrum
from ase import Atoms
from gpaw import GPAW, LCAO
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.mpi import world

xc = 'oldLDA'
c = +1
h = 0.4
b = 'dzp'
sy = 'Na2'
positions = [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]
atoms = Atoms(symbols=sy, positions=positions)
atoms.center(vacuum=3)

# LCAO-RT-TDDFT
calc = GPAW(mode=LCAO(force_complex_dtype=True),
            nbands=1,
            xc=xc,
            h=h,
            basis=b,
            charge=c,
            width=0,
            convergence={'density': 1e-8},
            setups={'Na': '1'})
atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('Na2.gpw', 'all')
del calc

calc = LCAOTDDFT('Na2.gpw')
dmfile = sy + '_lcao_restart_' + b + '_rt_z.dm' + str(world.size)
specfile = sy + '_lcao_restart_' + b + '_rt_z.spectrum' + str(world.size)
calc.absorption_kick([0.0, 0, 0.001])
calc.propagate(10, 20, dmfile)
if world.rank == 0:
    photoabsorption_spectrum(dmfile, specfile)

