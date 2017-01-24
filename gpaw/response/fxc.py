# -*- coding: utf-8
"""This module calculates XC kernels for the response model.
"""
from __future__ import print_function

import numpy as np
from gpaw.xc import XC
from gpaw.sphere.lebedev import weight_n, R_nv
from gpaw.mpi import world, rank, size
from gpaw.io.tar import Reader
from ase.utils.timing import Timer
from ase.units import Bohr, Ha


def get_xc_kernel(pd, chi0, functional='ALDA', chi0_wGG=None):
    """Factory function that calls the relevant functions below"""

    calc = chi0.calc
    fd = chi0.fd
    nspins = len(calc.density.nt_sG)
    assert nspins == 1

    if functional[0] == 'A':
        # Standard adiabatic kernel
        print('Calculating %s kernel' % functional, file=fd)
        Kxc_sGG = calculate_Kxc(pd, calc, functional=functional)
    elif functional[0] == 'r':
        # Renormalized kernel
        print('Calculating %s kernel' % functional, file=fd)
        from gpaw.xc.fxc import KernelDens
        kernel = KernelDens(calc,
                            functional,
                            [pd.kd.bzk_kc[0]],
                            fd,
                            calc.wfs.kd.N_c,
                            None,
                            ecut=pd.ecut * Ha,
                            tag='',
                            timer=Timer(),
                            )
        kernel.calculate_fhxc()
        r = Reader('fhxc_%s_%s_%s_%s.gpw' %
                   ('', functional, pd.ecut * Ha, 0))
        Kxc_sGG = np.array([r.get('fhxc_sGsG')])
        
        v_G = 4 * np.pi / pd.G2_qG[0]
        Kxc_sGG[0] -= np.diagflat(v_G)

        if pd.kd.gamma:
            Kxc_sGG[:, 0, :] = 0.0
            Kxc_sGG[:, :, 0] = 0.0
    elif functional[:2] == 'LR':
        print('Calculating LR kernel with alpha = %s' % functional[2:],
              file=fd)
        Kxc_sGG = calculate_lr_kernel(pd, calc,
                                      alpha=float(functional[2:]))
    elif functional == 'DM':
        print('Calculating DM kernel', file=fd)
        Kxc_sGG = calculate_dm_kernel(pd, calc)
    elif functional == 'Bootstrap':
        print('Calculating Bootstrap kernel', file=fd)
        if chi0.world.rank == 0:
            chi0_GG = chi0_wGG[0]
            chi0.world.broadcast(chi0_GG, 0)
        else:
            nG = pd.ngmax
            chi0_GG = np.zeros((nG, nG), complex)
            chi0.world.broadcast(chi0_GG, 0)
        Kxc_sGG = calculate_bootstrap_kernel(pd, chi0_GG, fd)
    else:
        raise ValueError('%s kernel not implemented' % functional)

    return Kxc_sGG


def calculate_Kxc(pd, calc, functional='ALDA', density_cut=None):
    """ALDA kernel"""
    
    gd = pd.gd
    npw = pd.ngmax
    nG = pd.gd.N_c
    vol = pd.gd.volume
    G_Gv = pd.get_reciprocal_vectors()
    nt_sG = calc.density.nt_sG
    R_av = calc.atoms.positions / Bohr
    setups = calc.wfs.setups
    D_asp = calc.density.D_asp

    # The soft part
    # assert np.abs(nt_sG[0].shape - nG).sum() == 0
    if functional == 'ALDA_X':
        x_only = True
        A_x = -3. / 4. * (3. / np.pi)**(1. / 3.)
        nspins = len(nt_sG)
        assert nspins in [1, 2]
        fxc_sg = nspins**(1. / 3.) * 4. / 9. * A_x * nt_sG**(-2. / 3.)
    else:
        assert len(nt_sG) == 1
        x_only = False
        fxc_sg = np.zeros_like(nt_sG)
        xc = XC(functional[1:])
        xc.calculate_fxc(gd, nt_sG, fxc_sg)

    if density_cut is not None:
        fxc_sg[np.where(nt_sG * len(nt_sG) < density_cut)] = 0.0

    # FFT fxc(r)
    nG0 = nG[0] * nG[1] * nG[2]
    tmp_sg = [np.fft.fftn(fxc_sg[s]) * vol / nG0 for s in range(len(nt_sG))]

    Kxc_sGG = np.zeros((len(fxc_sg), npw, npw), dtype=complex)
    for s in range(len(fxc_sg)):
        for iG, iQ in enumerate(pd.Q_qG[0]):
            iQ_c = (np.unravel_index(iQ, nG) + nG // 2) % nG - nG // 2
            for jG, jQ in enumerate(pd.Q_qG[0]):
                jQ_c = (np.unravel_index(jQ, nG) + nG // 2) % nG - nG // 2
                ijQ_c = (iQ_c - jQ_c)
                if (abs(ijQ_c) < nG // 2).all():
                    Kxc_sGG[s, iG, jG] = tmp_sg[s][tuple(ijQ_c)]

    # The PAW part
    KxcPAW_sGG = np.zeros_like(Kxc_sGG)
    dG_GGv = np.zeros((npw, npw, 3))
    for v in range(3):
        dG_GGv[:, :, v] = np.subtract.outer(G_Gv[:, v], G_Gv[:, v])

    for a, setup in enumerate(setups):
        if rank == a % size:
            rgd = setup.xc_correction.rgd
            n_qg = setup.xc_correction.n_qg
            nt_qg = setup.xc_correction.nt_qg
            nc_g = setup.xc_correction.nc_g
            nct_g = setup.xc_correction.nct_g
            Y_nL = setup.xc_correction.Y_nL
            dv_g = rgd.dv_g

            D_sp = D_asp[a]
            B_pqL = setup.xc_correction.B_pqL
            D_sLq = np.inner(D_sp, B_pqL.T)
            nspins = len(D_sp)

            f_sg = rgd.empty(nspins)
            ft_sg = rgd.empty(nspins)

            n_sLg = np.dot(D_sLq, n_qg)
            nt_sLg = np.dot(D_sLq, nt_qg)

            # Add core density
            n_sLg[:, 0] += np.sqrt(4. * np.pi) / nspins * nc_g
            nt_sLg[:, 0] += np.sqrt(4. * np.pi) / nspins * nct_g

            coefatoms_GG = np.exp(-1j * np.inner(dG_GGv, R_av[a]))
            for n, Y_L in enumerate(Y_nL):
                w = weight_n[n]
                f_sg[:] = 0.0
                n_sg = np.dot(Y_L, n_sLg)
                if x_only:
                    f_sg = nspins * (4 / 9.) * A_x * (nspins * n_sg)**(-2 / 3.)
                else:
                    xc.calculate_fxc(rgd, n_sg, f_sg)

                ft_sg[:] = 0.0
                nt_sg = np.dot(Y_L, nt_sLg)
                if x_only:
                    ft_sg = nspins * (4 / 9.) * (A_x
                                                 * (nspins * nt_sg)**(-2 / 3.))
                else:
                    xc.calculate_fxc(rgd, nt_sg, ft_sg)
                for i in range(len(rgd.r_g)):
                    coef_GG = np.exp(-1j * np.inner(dG_GGv, R_nv[n])
                                     * rgd.r_g[i])
                    for s in range(len(f_sg)):
                        KxcPAW_sGG[s] += w * np.dot(coef_GG,
                                                    (f_sg[s, i] -
                                                     ft_sg[s, i])
                                                    * dv_g[i]) * coefatoms_GG

    world.sum(KxcPAW_sGG)
    Kxc_sGG += KxcPAW_sGG

    if pd.kd.gamma:
        Kxc_sGG[:, 0, :] = 0.0
        Kxc_sGG[:, :, 0] = 0.0

    return Kxc_sGG / vol

    
def calculate_lr_kernel(pd, calc, alpha=0.2):
    """Long range kernel: fxc = \alpha / |q+G|^2"""
    
    assert pd.kd.gamma

    f_G = np.zeros(len(pd.G2_qG[0]))
    f_G[0] = -alpha
    f_G[1:] = -alpha / pd.G2_qG[0][1:]

    return np.array([np.diag(f_G)])
    
    
def calculate_dm_kernel(pd, calc):
    """Density matrix kernel"""
    
    assert pd.kd.gamma
    
    nv = calc.wfs.setups.nvalence
    psit_nG = np.array([calc.wfs.kpt_u[0].psit_nG[n]
                        for n in range(4 * nv)])
    vol = np.linalg.det(calc.wfs.gd.cell_cv)
    Ng = np.prod(calc.wfs.gd.N_c)
    rho_GG = np.dot(psit_nG.conj().T, psit_nG) * vol / Ng**2

    maxG2 = np.max(pd.G2_qG[0])
    cut_G = np.arange(calc.wfs.pd.ngmax)[calc.wfs.pd.G2_qG[0] <= maxG2]
    
    G_G = pd.G2_qG[0]**0.5
    G_G[0] = 1.0
    
    Kxc_GG = np.diagflat(4 * np.pi / G_G**2)
    Kxc_GG = np.dot(Kxc_GG, rho_GG.take(cut_G, 0).take(cut_G, 1))
    Kxc_GG -= 4 * np.pi * np.diagflat(1.0 / G_G**2)

    return np.array([Kxc_GG])
    

def calculate_bootstrap_kernel(pd, chi0_GG, fd):
    """Bootstrap kernel PRL 107, 186401"""

    if pd.kd.gamma:
        v_G = np.zeros(len(pd.G2_qG[0]))
        v_G[0] = 4 * np.pi
        v_G[1:] = 4 * np.pi / pd.G2_qG[0][1:]
    else:
        v_G = 4 * np.pi / pd.G2_qG[0]

    nG = len(v_G)
    K_GG = np.diag(v_G)

    fxc_GG = np.zeros((nG, nG), dtype=complex)
    dminv_GG = np.zeros((nG, nG), dtype=complex)
                
    for iscf in range(120):
        dminvold_GG = dminv_GG.copy()
        Kxc_GG = K_GG + fxc_GG

        chi_GG = np.dot(np.linalg.inv(np.eye(nG, nG)
                                      - np.dot(chi0_GG, Kxc_GG)), chi0_GG)
        dminv_GG = np.eye(nG, nG) + np.dot(K_GG, chi_GG)

        alpha = dminv_GG[0, 0] / (K_GG[0, 0] * chi0_GG[0, 0])
        fxc_GG = alpha * K_GG
        print(iscf, 'alpha =', alpha, file=fd)
        error = np.abs(dminvold_GG - dminv_GG).sum()
        if np.sum(error) < 0.1:
            print('Self consistent fxc finished in %d iterations !' % iscf,
                  file=fd)
            break
        if iscf > 100:
            print('Too many fxc scf steps !', file=fd)
    
    return np.array([fxc_GG])
