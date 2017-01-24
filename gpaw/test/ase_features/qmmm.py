from math import cos, sin

from ase import Atoms
from ase.calculators.tip3p import (TIP3P, epsilon0, sigma0, rOH, thetaHOH,
                                   set_tip3p_charges)
from ase.calculators.qmmm import EIQMMM, LJInteractions, Embedding
from ase.constraints import FixInternals
from ase.optimize import BFGS
from gpaw import GPAW

r = rOH
a = thetaHOH

interaction = LJInteractions({('O', 'O'): (epsilon0, sigma0)})

for selection in [[0, 1, 2], [3, 4, 5]]:
    name = ''.join(str(i) for i in selection)
    dimer = Atoms('H2OH2O',
                  [(-r * cos(a / 2), r * sin(a / 2), 0),
                   (-r * cos(a / 2), -r * sin(a / 2), 0),
                   (0, 0, 0),
                   (-r * cos(a), 0, r * sin(a)),
                   (-r, 0, 0),
                   (0, 0, 0),
                   ])
    dimer.positions[3:, 0] += 2.8
    dimer.constraints = FixInternals(
        bonds=[(r, (0, 2)), (r, (1, 2)),
               (r, (3, 5)), (r, (4, 5))],
        angles=[(a, (0, 2, 1)), (a, (3, 5, 4))])

    set_tip3p_charges(dimer)
    dimer.calc = EIQMMM(selection,
                        GPAW(txt=name + '.txt', h=0.17),
                        TIP3P(),
                        interaction,
                        vacuum=4,
                        embedding=Embedding(rc=0.2),
                        output=name + '.out')
    print(dimer.get_potential_energy())
    opt = BFGS(dimer, trajectory=name + '.traj')
    opt.run(0.02)

    c = dimer.constraints
    dimer.constraints = None
    monomer = dimer[selection]
    dimer.constraints = c
    monomer.center(vacuum=4)
    monomer.calc = GPAW(txt=name + 'M.txt', h=0.17)
    e0 = monomer.get_potential_energy()
    be = dimer.get_potential_energy() - e0
    d = dimer.get_distance(2, 5)
    print(name, be, d)
    if name == '012':
        assert abs(be - -0.260) < 0.002
        assert abs(d - 2.79) < 0.02
    else:
        assert abs(be - -0.346) < 0.002
        assert abs(d - 2.67) < 0.02
    opt = BFGS(monomer, trajectory=name + 'M.traj')
    opt.run(0.02)
