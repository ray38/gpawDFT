import time

import ase.db
import ase.optimize
from ase.collections import g2

from gpaw import GPAW, Mixer


optimizers = ['BFGS', 'BFGSLineSearch',
              'FIRE', 'GoodOldQuasiNewton',
              'LBFGS', 'LBFGSLineSearch']

con = ase.db.connect('g2_dzp.db')
for name, atoms in zip(g2.names, g2):
    if len(atoms) == 1:
        continue
    atoms.center(vacuum=3.5)
    for optimizer in optimizers:
        id = con.reserve(name=name, optimizer=optimizer)
        if id is None:
            continue
        mol = atoms.copy()
        mol.calc = GPAW(mode='lcao',
                        basis='dzp',
                        h=0.17,
                        xc='PBE',
                        mixer=Mixer(0.05, 2),
                        txt='{0}-{1}.txt'.format(name, optimizer))
        Optimizer = getattr(ase.optimize, optimizer)
        opt = Optimizer(mol)
        t = time.time()
        try:
            opt.run(fmax=0.05, steps=50)
        except Exception as ex:
            print(name, optimizer, ex)
            continue
        con.write(mol, name=name, optimizer=optimizer,
                  steps=opt.nsteps, time=time.time() - t)
        del con[id]
