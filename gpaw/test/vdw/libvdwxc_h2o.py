from ase.build import molecule
from gpaw import GPAW, Mixer
from gpaw.atom.generator import Generator
from gpaw.xc.libvdwxc import vdw_df, vdw_mbeef

system = molecule('H2O')
system.center(vacuum=1.5)
system.pbc = 1

for mode in ['lcao', 'fd', 'pw']:
    kwargs = dict(mode=mode,
                  basis='szp(dzp)',
                  xc=vdw_df(),
                  mixer=Mixer(0.3, 5, 10.))
    calc = GPAW(**kwargs)
    def stopcalc():
        calc.scf.converged = True
    calc.attach(stopcalc, 6)

    system.set_calculator(calc)
    system.get_potential_energy()
