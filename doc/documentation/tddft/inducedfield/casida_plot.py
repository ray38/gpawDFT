# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from gpaw.inducedfield.inducedfield_base import BaseInducedField
from gpaw.tddft.units import (attosec_to_autime, autime_to_attosec,
                              eV_to_aufrequency, aufrequency_to_eV)


# Helper function
def do_plot(d_g, ng, box, atoms):
    # Take slice of data array
    d_yx = d_g[:, ng[1] // 2, :]
    y = np.linspace(0, box[0], ng[0])
    ylabel = u'x / Å'
    x = np.linspace(0, box[2], ng[2])
    xlabel = u'z / Å'
    
    # Plot
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    X, Y = np.meshgrid(x, y)
    plt.contourf(X, Y, d_yx, 40)
    plt.colorbar()
    for atom in atoms:
        pos = atom.position
        plt.scatter(pos[2], pos[0], s=50, c='k', marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([x[0], x[-1]])
    plt.ylim([y[0], y[-1]])
    ax.set_aspect('equal')


# Read InducedField object
ind = BaseInducedField('na2_casida_field.ind', readmode='all')

# Choose array
w = 1                                      # Frequency index
freq = ind.omega_w[w] * aufrequency_to_eV  # Frequency
box = np.diag(ind.atoms.get_cell())        # Calculation box
d_g = ind.Ffe_wg[w]                        # Data array
ng = d_g.shape                             # Size of grid
atoms = ind.atoms                          # Atoms

do_plot(d_g, ng, box, atoms)
plt.title('Field enhancement @ %.2f eV' % freq)
plt.savefig('na2_casida_Ffe.png', bbox_inches='tight')

# Imaginary part of density
d_g = ind.Frho_wg[w].imag * 1e-3  # Multiply by kick strength
ng = d_g.shape
do_plot(d_g, ng, box, atoms)
plt.title('Imaginary part of induced charge density @ %.2f eV' % freq)
plt.savefig('na2_casida_Frho.png', bbox_inches='tight')

# Imaginary part of potential
d_g = ind.Fphi_wg[w].imag * 1e-3  # Multiply by kick strength
ng = d_g.shape
do_plot(d_g, ng, box, atoms)
plt.title('Imaginary part of induced potential @ %.2f eV' % freq)
plt.savefig('na2_casida_Fphi.png', bbox_inches='tight')
