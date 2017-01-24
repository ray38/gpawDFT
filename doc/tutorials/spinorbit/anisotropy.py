from __future__ import print_function
import numpy as np
from gpaw import GPAW
from gpaw.spinorbit import get_spinorbit_eigenvalues, set_calculator

calc = GPAW('gs_Co.gpw', txt=None)

e_skn = np.array([[calc.get_eigenvalues(kpt=k, spin=s)
                   for k in range(len(calc.get_ibz_k_points()))]
                  for s in range(2)])
e_kn = np.reshape(np.swapaxes(e_skn, 0, 1),
                  (len(e_skn[0]), 2 * len(e_skn[0, 0])))
e_kn = np.sort(e_kn, 1)
f_skn = np.array([[calc.get_occupation_numbers(kpt=k, spin=s)
                  for k in range(len(calc.get_ibz_k_points()))]
                 for s in range(2)])
f_kn = np.reshape(np.swapaxes(f_skn, 0, 1),
                  (len(f_skn[0]), 2 * len(f_skn[0, 0])))
f_kn = np.sort(f_kn, 1)[:, ::-1]
E = np.sum(e_kn * f_kn)

# Spinorbit
theta_i = [i * np.pi / 20 for i in range(21)]
for theta in theta_i:
    calc = GPAW('gs_Co.gpw', txt=None)
    e_mk = get_spinorbit_eigenvalues(calc, theta=theta, phi=0.0)
    set_calculator(calc, e_mk.T)
    f_km = np.array([calc.get_occupation_numbers(kpt=k)
                     for k in range(len(calc.get_ibz_k_points()))])
    
    E_so = np.sum(e_mk.T * f_km)

    with open('anisotropy.dat', 'a') as f:
        print(theta, E, E_so, file=f)
