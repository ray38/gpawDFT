from math import pi, sqrt

import numpy as np
from ase.units import Bohr, Hartree

from gpaw.fftw import get_efficient_fft_size
from gpaw.grid_descriptor import GridDescriptor
from gpaw.lfc import LFC
from gpaw.utilities import h2gpts
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.mpi import serial_comm


class Interpolator:
    def __init__(self, gd1, gd2, dtype=float):
        self.pd1 = PWDescriptor(0.0, gd1, dtype)
        self.pd2 = PWDescriptor(0.0, gd2, dtype)

    def interpolate(self, a_r):
        return self.pd1.interpolate(a_r, self.pd2)[0]

        
POINTS = 200


def gauss(rgd, alpha):
    r_g = rgd.r_g
    g_g = 4 / sqrt(pi) * alpha**1.5 * np.exp(-alpha * r_g**2)
    return g_g
    

class PS2AE:
    """Transform PS to AE wave functions.
    
    Interpolates PS wave functions to a fine grid and adds PAW
    corrections in order to obtain true AE wave functions.
    """
    def __init__(self, calc, h=0.05, n=2):
        """Create transformation object.
        
        calc: GPAW calculator object
            The calcalator that has the wave functions.
        h: float
            Desired grid-spacing in Angstrom.
        n: int
            Force number of points to be a mulitiple of n.
        """
        self.calc = calc
        gd = calc.wfs.gd

        gd1 = GridDescriptor(gd.N_c, gd.cell_cv, comm=serial_comm)
        
        # Descriptor for the final grid:
        N_c = h2gpts(h / Bohr, gd.cell_cv)
        N_c = np.array([get_efficient_fft_size(N, n) for N in N_c])
        gd2 = self.gd = GridDescriptor(N_c, gd.cell_cv, comm=serial_comm)
        self.interpolator = Interpolator(gd1, gd2, self.calc.wfs.dtype)

        self.dphi = None  # PAW correction (will be initialize when needed)

    def _initialize_corrections(self):
        if self.dphi is not None:
            return
        splines = {}
        dphi_aj = []
        for setup in self.calc.wfs.setups:
            dphi_j = splines.get(setup)
            if dphi_j is None:
                rcut = max(setup.rcut_j) * 1.1
                gcut = setup.rgd.ceil(rcut)
                dphi_j = []
                for l, phi_g, phit_g in zip(setup.l_j,
                                            setup.data.phi_jg,
                                            setup.data.phit_jg):
                    dphi_g = (phi_g - phit_g)[:gcut]
                    dphi_j.append(setup.rgd.spline(dphi_g, rcut, l,
                                                   points=200))
            dphi_aj.append(dphi_j)
            
        self.dphi = LFC(self.gd, dphi_aj, kd=self.calc.wfs.kd.copy(),
                        dtype=self.calc.wfs.dtype)
        self.dphi.set_positions(self.calc.atoms.get_scaled_positions())
        
    def get_wave_function(self, n, k=0, s=0, ae=True):
        """Interpolate wave function.
        
        n: int
            Band index.
        k: int
            K-point index.
        s: int
            Spin index.
        ae: bool
            Add PAW correction to get an all-electron wave function.
        """
        psi_r = self.calc.get_pseudo_wave_function(n, k, s,
                                                   pad=True, periodic=True)
        psi_R = self.interpolator.interpolate(psi_r * Bohr**1.5)
        if ae:
            self._initialize_corrections()
            wfs = self.calc.wfs
            P_nI = wfs.collect_projections(k, s)
            if wfs.world.rank == 0:
                P_ai = {}
                I1 = 0
                for a, setup in enumerate(wfs.setups):
                    I2 = I1 + setup.ni
                    P_ai[a] = P_nI[n, I1:I2]
                    I1 = I2
                self.dphi.add(psi_R, P_ai, k)
            wfs.world.broadcast(psi_R, 0)
        return psi_R
        
    def get_electrostatic_potential(self, ae=True, rcgauss=0.02):
        ham = self.calc.hamiltonian

        if ham.vHt_g is None:
            self.calc.restore_state()
            
        gd = ham.finegd
        v_r = gd.zero_pad(ham.vHt_g)
        gd1 = GridDescriptor(gd.N_c, gd.cell_cv, comm=serial_comm)
        interpolator = Interpolator(gd1, self.gd)
        v_R = interpolator.interpolate(v_r)

        if ae:
            alpha = 1 / (rcgauss / Bohr)**2
            self.add_potential_correction(v_R, alpha)
            
        return v_R * Hartree

    def add_potential_correction(self, v_R, alpha):
        dens = self.calc.density
        dv_a1 = []
        for a, D_sp in dens.D_asp.items():
            setup = dens.setups[a]
            c = setup.xc_correction
            rgd = c.rgd
            ghat_g = gauss(rgd, 1 / setup.rcgauss**2)
            Z_g = gauss(rgd, alpha) * setup.Z
            D_q = np.dot(D_sp.sum(0), c.B_pqL[:, :, 0])
            dn_g = np.dot(D_q, (c.n_qg - c.nt_qg)) * sqrt(4 * pi)
            dn_g += 4 * pi * (c.nc_g - c.nct_g)
            dn_g -= Z_g
            dn_g -= dens.Q_aL[a][0] * ghat_g * sqrt(4 * pi)
            dv_g = rgd.poisson(dn_g) / sqrt(4 * pi)
            dv_g[1:] /= rgd.r_g[1:]
            dv_g[0] = dv_g[1]
            dv_g[-1] = 0.0
            dv_a1.append([rgd.spline(dv_g, points=POINTS)])
            
        dv = LFC(self.gd, dv_a1)
        dv.set_positions(self.calc.atoms.get_scaled_positions())
        dv.add(v_R)
