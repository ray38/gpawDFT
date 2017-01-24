import numpy as np
from ase import Atoms
from gpaw import GPAW
from gpaw.cdft import CDFT

d0 = 2.5
he2 = Atoms('He2', positions=([0, 0, 0], [0, 0, d0]))
he2.center(4)
calc = GPAW(charge=1, xc='PBE', txt='he2.txt')
he2.calc = CDFT(calc, [[0]], [1], txt='he2-cdft.txt', tolerance=0.001)
for d in np.linspace(d0, d0 + 0.1, 6):
    he2.set_distance(0, 1, d, 0)
    e = he2.get_potential_energy()
    f = he2.get_forces()
    print(he2.positions[:, 2], e, f[:, 2])
