"""This module defines different external potentials."""
import warnings

import numpy as np

from ase.units import Bohr, Hartree

import _gpaw

__all__ = ['ConstantPotential', 'ConstantElectricField']


def create_external_potential(name, **kwargs):
    """Construct potential from dict."""
    if name not in __all__:
        raise ValueError
    return globals()[name](**kwargs)


class ExternalPotential:
    vext_g = None

    def get_potential(self, gd):
        """Get the potential on a regular 3-d grid.

        Will only call calculate_potential() the first time."""

        if self.vext_g is None:
            self.calculate_potential(gd)
        return self.vext_g

    def calculate_potential(self, gd):
        raise NotImplementedError


class ConstantPotential(ExternalPotential):
    """Constant potential for tests."""
    def __init__(self, constant=1.0):
        self.constant = constant / Hartree

    def __str__(self):
        return 'Constant potential: {0:.3f} eV'.format(self.constant * Hartree)

    def calculate_potential(self, gd):
        self.vext_g = gd.zeros() + self.constant

    def todict(self):
        return {'name': 'ConstantPotential',
                'constant': self.constant * Hartree}


class ConstantElectricField(ExternalPotential):
    def __init__(self, strength, direction=[0, 0, 1], tolerance=1e-7):
        """External constant electric field.

        strength: float
            Field strength in V/Ang.
        direction: vector
            Polarisation direction.
        """
        d_v = np.asarray(direction)
        self.field_v = strength * d_v / (d_v**2).sum()**0.5 * Bohr / Hartree
        self.tolerance = tolerance

    def __str__(self):
        return ('Constant electric field: '
                '({0:.3f}, {1:.3f}, {2:.3f}) eV/Ang'
                .format(*(self.field_v * Hartree / Bohr)))

    def calculate_potential(self, gd):
        d_v = self.field_v / (self.field_v**2).sum()**0.5
        for axis_v in gd.cell_cv[gd.pbc_c]:
            if abs(np.dot(d_v, axis_v)) > self.tolerance:
                raise ValueError(
                    'Field not perpendicular to periodic axis: {0}'
                    .format(axis_v))

        center_v = 0.5 * gd.cell_cv.sum(0)
        r_gv = gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
        self.vext_g = np.dot(r_gv - center_v, self.field_v)

    def todict(self):
        strength = (self.field_v**2).sum()**0.5
        return {'name': 'ConstantElectricField',
                'strength': strength * Hartree / Bohr,
                'direction': self.field_v / strength}


class PointChargePotential(ExternalPotential):
    def __init__(self, charges, positions=None,
                 rc=0.2, rc2=np.inf, width=1.0):
        """Point-charge potential.

        charges: list of float
            Charges.
        positions: (N, 3)-shaped array-like of float
            Positions of charges in Angstrom.  Can be set later.
        rc: float
            Inner cutoff for Coulomb potential in Angstrom.
        rc2: float
            Outer cutoff for Coulomb potential in Angstrom.
        width: float
            Width for cutoff function for Coulomb part.

        For r < rc, 1 / r is replaced by a third order polynomial in r^2 that
        has matching value, first derivative, second derivative and integral.

        For rc2 - width < r < rc2, 1 / r is multiplied by a smooth cutoff
        function (a third order polynomium in r).

        You can also give rc a negative value.  In that case, this formula
        is used::

            (r^4 - rc^4) / (r^5 - |rc|^5)

        for all values of r - no cutoff at rc2!
        """
        self.q_p = np.ascontiguousarray(charges, float)
        self.rc = rc / Bohr
        self.rc2 = rc2 / Bohr
        self.width = width / Bohr
        if positions is not None:
            self.set_positions(positions)
        else:
            self.R_pv = None

        if abs(self.q_p).max() < 1e-14:
            warnings.warn('No charges!')

    def __str__(self):
        return ('Point-charge potential '
                '(points: {0}, cutoffs: {1:.3f}, {2:.3f}, {3:.3f} Ang)'
                .format(len(self.q_p),
                        self.rc * Bohr,
                        (self.rc2 - self.width) * Bohr,
                        self.rc2 * Bohr))

    def set_positions(self, R_pv):
        """Update positions."""
        self.R_pv = np.asarray(R_pv) / Bohr
        self.vext_g = None

    def calculate_potential(self, gd):
        assert gd.orthogonal
        self.vext_g = gd.zeros()
        _gpaw.pc_potential(gd.beg_c, gd.h_cv.diagonal().copy(),
                           self.q_p, self.R_pv, self.rc, self.rc2, self.width,
                           self.vext_g)

    def get_forces(self, calc):
        """Calculate forces from QM charge density on point-charges."""
        dens = calc.density
        F_pv = np.zeros_like(self.R_pv)
        gd = dens.finegd
        _gpaw.pc_potential(gd.beg_c, gd.h_cv.diagonal().copy(),
                           self.q_p, self.R_pv, self.rc, self.rc2, self.width,
                           self.vext_g, dens.rhot_g, F_pv)
        gd.comm.sum(F_pv)
        return F_pv * Hartree / Bohr
