from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from gpaw.response.gw_bands import GWBands


# Initializing bands object
K = np.array([1 / 3, 1 / 3, 0])
G = np.array([0.0, 0.0, 0.0])
kpoints = np.array([G, K, G])

GW = GWBands(calc='MoS2_fulldiag.gpw',
             gw_file='MoS2_g0w0_80_results.pckl',
             kpoints=kpoints)

# Without spin-orbit
results = GW.get_gw_bands(SO=False, interpolate=True, vac=True)

x_x = results['x_k']
X = results['X']
eGW_kn = results['e_kn']
ef = results['ef']


# Plotting Bands
labels_K = [r'$\Gamma$', r'$K$', r'$\Gamma$']

f = plt.figure()
plt.plot(x_x, eGW_kn, '-r')

plt.axhline(ef, color='k', linestyle='--')

for p in X:
    plt.axvline(p, color='k')

plt.xlim(0, x_x[-1])
plt.xticks(X, labels_K, fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('E - E$_{vac}$ (eV)', fontsize=24)
plt.savefig('MoS2_bs.png')
plt.show()
