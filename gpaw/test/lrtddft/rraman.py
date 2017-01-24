import sys
from ase import Atoms, Atom
from ase.vibrations.resonant_raman import ResonantRaman

from gpaw import GPAW, FermiDirac
from gpaw.cluster import Cluster
from gpaw.lrtddft.kssingle import KSSingles
from gpaw.mpi import world

txt = '-'
txt = None
load = True
load = False
xc = 'LDA'

R = 0.7  # approx. experimental bond length
a = 4.0
c = 5.0
H2 = Atoms([Atom('H', (a / 2, a / 2, (c - R) / 2)),
            Atom('H', (a / 2, a / 2, (c + R) / 2))],
           cell=(a, a, c))

calc = GPAW(xc=xc, nbands=3, spinpol=False, eigensolver='rmm-diis', txt=txt)
H2.set_calculator(calc)
H2.get_potential_energy()

gsname = exname = 'rraman'
rr = ResonantRaman(H2, KSSingles, gsname=gsname, exname=exname,
                   exkwargs={'eps':0.0, 'jend':1}
               )
rr.run()

# check size
kss = KSSingles('rraman-d0.010.eq.ex.gz')
assert(len(kss) == 1)

rr = ResonantRaman(H2, KSSingles, gsname=gsname, exname=exname, 
                   verbose=True,)
rr.summary(omega=5, method='frederiksen')
