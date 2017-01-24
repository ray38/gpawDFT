from ase import Atoms
from gpaw import GPAW
from gpaw.mixer import FFTMixer
from gpaw.wavefunctions.pw import PW
from gpaw.test import equal

bulk = Atoms('Li', pbc=True)
bulk.set_cell((2.6, 2.6, 2.6))
k = 4
calc = GPAW(mode=PW(200),
            kpts=(k, k, k),
            mixer=FFTMixer(),
            verbose=1,
            eigensolver='rmm-diis')
bulk.set_calculator(calc)
e = bulk.get_potential_energy()
equal(e, -1.98481281259, 1.0e-6)
