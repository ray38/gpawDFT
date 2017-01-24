from __future__ import print_function
from gpaw import GPAW
from ase.build import molecule
from gpaw.mpi import world

system = molecule('H2O')
system.cell = (4, 4, 4)
system.pbc = 1


for mode in ['fd',
             'pw',
             'lcao'
             ]:
    energy = []

    if mode != 'lcao':
        eigensolver = 'rmm-diis'
    else:
        eigensolver = None

    for augment_grids in 0, 1:
        if world.rank == 0:
            print(mode, augment_grids)
        calc = GPAW(mode=mode,
                    gpts=(16, 16, 16),
                    txt='gpaw.%s.%d.txt' % (mode, int(augment_grids)),
                    eigensolver=eigensolver,
                    parallel=dict(augment_grids=augment_grids),
                    basis='szp(dzp)',
                    kpts=[1, 1, 4],
                    nbands=8)
        def stopcalc():
            calc.scf.converged = True
        # Iterate enough for density to update so it depends on potential
        calc.attach(stopcalc, 3 if mode == 'lcao' else 5)
        system.set_calculator(calc)
        energy.append(system.get_potential_energy())
    err = energy[1] - energy[0]
    assert err < 1e-10, err
