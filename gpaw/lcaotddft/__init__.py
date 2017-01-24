from __future__ import print_function
from math import log as ln

import numpy as np
from numpy.linalg import inv, solve
from ase.units import Bohr, Hartree

from gpaw import GPAW
from gpaw.external import ConstantElectricField
from gpaw.utilities.blas import gemm
from gpaw.mixer import DummyMixer
from gpaw.tddft.units import attosec_to_autime, autime_to_attosec
from gpaw.xc import XC

from gpaw.utilities.scalapack import (pblas_simple_hemm, pblas_simple_gemm,
                                      scalapack_inverse, scalapack_solve,
                                      scalapack_zero, pblas_tran,
                                      scalapack_set)
                                     
from time import localtime


class KickHamiltonian:
    def __init__(self, calc, ext):
        ham = calc.hamiltonian
        dens = calc.density
        vext_g = ext.get_potential(ham.finegd)
        self.vt_sG = [ham.restrict_and_collect(vext_g)]
        self.dH_asp = ham.setups.empty_atomic_matrix(1, ham.atom_partition)

        W_aL = dens.ghat.dict()
        dens.ghat.integrate(vext_g, W_aL)
        # XXX this is a quick hack to get the distribution right
        dHtmp_asp = ham.atomdist.to_aux(self.dH_asp)
        for a, W_L in W_aL.items():
            setup = dens.setups[a]
            dHtmp_asp[a] = np.dot(setup.Delta_pL, W_L).reshape((1, -1))
        self.dH_asp = ham.atomdist.from_aux(dHtmp_asp)

        
class LCAOTDDFT(GPAW):
    def __init__(self, filename=None,
                 propagator='cn', fxc=None, **kwargs):
        self.time = 0.0
        self.niter = 0
        self.kick_strength = np.zeros(3)
        self.tddft_initialized = False
        self.fxc = fxc
        self.propagator = propagator
        if filename is None:
            kwargs['mode'] = kwargs.get('mode', 'lcao')
        GPAW.__init__(self, filename, **kwargs)

        # Restarting from a file
        if filename is not None:
            #self.initialize()
            self.set_positions()
            
    def read(self, filename):
        reader = GPAW.read(self, filename)
        if 'tddft' in reader:
            self.time = reader.tddft.time
            self.niter = reader.tddft.niter
            self.kick_strength = reader.tddft.kick_strength

    def _write(self, writer, mode):
        GPAW._write(self, writer, mode)
        writer.child('tddft').write(time=self.time,
                                    niter=self.niter,
                                    kick_strength=self.kick_strength)
        
    def propagate_wfs(self, sourceC_nm, targetC_nm, S_MM, H_MM, dt):
        if self.propagator == 'cn':
            return self.linear_propagator(sourceC_nm, targetC_nm, S_MM, H_MM,
                                          dt)
        raise NotImplementedError

    def linear_propagator(self, sourceC_nM, targetC_nM, S_MM, H_MM, dt):
        self.timer.start('Linear solve')

        if self.blacs:
            # XXX, Preallocate
            target_blockC_nm = self.Cnm_block_descriptor.empty(dtype=complex)
            temp_blockC_nm = self.Cnm_block_descriptor.empty(dtype=complex)
            temp_block_mm = self.mm_block_descriptor.empty(dtype=complex)
            if self.density.gd.comm.rank != 0:
                # XXX Fake blacks nbands, nao, nbands, nao grid because some
                # weird asserts
                # (these are 0,x or x,0 arrays)
                sourceC_nM = self.CnM_unique_descriptor.zeros(dtype=complex)

            # 1. target = (S+0.5j*H*dt) * source
            # Wave functions to target
            self.CnM2nm.redistribute(sourceC_nM, temp_blockC_nm)

            # XXX It can't be this f'n hard to symmetrize a matrix (tri2full)
            # Remove upper diagonal
            scalapack_zero(self.mm_block_descriptor, H_MM, 'U')
            # Lower diagonal matrix:
            temp_block_mm[:] = S_MM - (0.5j * dt) * H_MM
            scalapack_set(self.mm_block_descriptor, temp_block_mm, 0, 0, 'U')
            # Note it's strictly lower diagonal matrix
            # Add transpose of H
            pblas_tran(-0.5j * dt, H_MM, 1.0, temp_block_mm,
                       self.mm_block_descriptor, self.mm_block_descriptor)
            # Add transpose of S
            pblas_tran(1.0, S_MM, 1.0, temp_block_mm,
                       self.mm_block_descriptor, self.mm_block_descriptor)

            pblas_simple_gemm(self.Cnm_block_descriptor,
                              self.mm_block_descriptor,
                              self.Cnm_block_descriptor,
                              temp_blockC_nm,
                              temp_block_mm,
                              target_blockC_nm)
            # 2. target = (S-0.5j*H*dt)^-1 * target
            # temp_block_mm[:] = S_MM + (0.5j*dt) * H_MM
            # XXX It can't be this f'n hard to symmetrize a matrix (tri2full)
            # Lower diagonal matrix:
            temp_block_mm[:] = S_MM + (0.5j * dt) * H_MM
            # Not it's stricly lower diagonal matrix:
            scalapack_set(self.mm_block_descriptor, temp_block_mm, 0, 0, 'U')
            # Add transpose of H:
            pblas_tran(+0.5j * dt, H_MM, 1.0, temp_block_mm,
                       self.mm_block_descriptor, self.mm_block_descriptor)
            # Add transpose of S
            pblas_tran(1.0, S_MM, 1.0, temp_block_mm,
                       self.mm_block_descriptor, self.mm_block_descriptor)

            scalapack_solve(self.mm_block_descriptor,
                            self.Cnm_block_descriptor,
                            temp_block_mm,
                            target_blockC_nm)

            if self.density.gd.comm.rank != 0:  # XXX is this correct?
                # XXX Fake blacks nbands, nao, nbands, nao grid because some
                # weird asserts
                # (these are 0,x or x,0 arrays)
                target = self.CnM_unique_descriptor.zeros(dtype=complex)
            else:
                target = targetC_nM
            self.Cnm2nM.redistribute(target_blockC_nm, target)
            self.density.gd.comm.broadcast(targetC_nM, 0)  # Is this required?
        else:
            # Note: The full equation is conjugated (therefore -+, not +-)
            targetC_nM[:] = \
                solve(S_MM - 0.5j * H_MM * dt,
                      np.dot(S_MM + 0.5j * H_MM * dt,
                             sourceC_nM.T.conjugate())).T.conjugate()
        
        self.timer.stop('Linear solve')

    def taylor_propagator(self, sourceC_nM, targetC_nM, S_MM, H_MM, dt):
        self.timer.start('Taylor propagator')

        if self.blacs:
            # XXX, Preallocate
            target_blockC_nm = self.Cnm_block_descriptor.empty(dtype=complex)
            if self.density.gd.comm.rank != 0:
                # XXX Fake blacks nbands, nao, nbands, nao grid because some
                # weird asserts
                # (these are 0,x or x,0 arrays)
                sourceC_nM = self.CnM_unique_descriptor.zeros(dtype=complex)

            # Zeroth order taylor to target
            self.CnM2nm.redistribute(sourceC_nM, target_blockC_nm)

            # XXX, preallocate, optimize use of temporal arrays
            temp_blockC_nm = target_blockC_nm.copy()
            temp2_blockC_nm = target_blockC_nm.copy()

            order = 4
            assert self.wfs.kd.comm.size == 1
            for n in range(order):
                # Multiply with hamiltonian
                pblas_simple_hemm(self.mm_block_descriptor,
                                  self.Cnm_block_descriptor,
                                  self.Cnm_block_descriptor,
                                  H_MM,
                                  temp_blockC_nm,
                                  temp2_blockC_nm, side='R')
                # XXX: replace with not simple gemm
                temp2_blockC_nm *= -1j * dt / (n + 1)
                # Multiply with inverse overlap
                pblas_simple_hemm(self.mm_block_descriptor,
                                  self.Cnm_block_descriptor,
                                  self.Cnm_block_descriptor,
                                  self.wfs.kpt_u[0].invS_MM,  # XXX
                                  temp2_blockC_nm,
                                  temp_blockC_nm, side='R')
                target_blockC_nm += temp_blockC_nm
            if self.density.gd.comm.rank != 0:  # Todo: Change to gd.rank
                # XXX Fake blacks nbands, nao, nbands, nao grid because
                # some weird asserts
                # (these are 0,x or x,0 arrays)
                target = self.CnM_unique_descriptor.zeros(dtype=complex)
            else:
                target = targetC_nM
            self.Cnm2nM.redistribute(target_blockC_nm, target)

            self.density.gd.comm.broadcast(targetC_nM, 0)
        else:
            assert self.wfs.kd.comm.size == 1
            if self.density.gd.comm.rank == 0:
                targetC_nM[:] = sourceC_nM[:]
                tempC_nM = sourceC_nM.copy()
                order = 4
                for n in range(order):
                    tempC_nM[:] = \
                        np.dot(self.wfs.kpt_u[0].invS,
                               np.dot(H_MM, 1j * dt / (n + 1) *
                                      tempC_nM.T.conjugate())).T.conjugate()
                    targetC_nM += tempC_nM
            self.density.gd.comm.broadcast(targetC_nM, 0)
                
        self.timer.stop('Taylor propagator')

    def absorption_kick(self, kick_strength):
        self.tddft_init()
        self.timer.start('Kick')
        self.kick_strength = np.array(kick_strength, dtype=float)

        # magnitude
        magnitude = np.sqrt(np.sum(self.kick_strength**2))

        # normalize
        direction = self.kick_strength / magnitude

        self.log('Applying absorption kick')
        self.log('Magnitude: %.8f hartree/bohr' % magnitude)
        self.log('Direction: %.4f %.4f %.4f' % tuple(direction))

        # Create hamiltonian object for absorption kick
        cef = ConstantElectricField(magnitude * Hartree / Bohr, direction)
        kick_hamiltonian = KickHamiltonian(self, cef)
        for k, kpt in enumerate(self.wfs.kpt_u):
            Vkick_MM = self.wfs.eigensolver.calculate_hamiltonian_matrix(
                kick_hamiltonian, self.wfs, kpt, add_kinetic=False, root=-1)
            for i in range(10):
                self.propagate_wfs(kpt.C_nM, kpt.C_nM, kpt.S_MM, Vkick_MM, 0.1)
        self.timer.stop('Kick')

    def blacs_mm_to_global(self, H_mm):
        # Someone could verify that this works and remove the error.
        raise NotImplementedError('Method untested and thus unreliable')
        target = self.MM_descriptor.empty(dtype=complex)
        self.mm2MM.redistribute(H_mm, target)
        self.wfs.world.barrier()
        return target

    def blacs_nm_to_global(self, C_nm):
        # Someone could verify that this works and remove the error.
        raise NotImplementedError('Method untested and thus unreliable')
        target = self.CnM_unique_descriptor.empty(dtype=complex)
        self.Cnm2nM.redistribute(C_nm, target)
        self.wfs.world.barrier()
        return target

    def tddft_init(self):
        if self.tddft_initialized:
            return
        self.blacs = self.wfs.ksl.using_blacs
        if self.blacs:
            self.ksl = ksl = self.wfs.ksl
            nao = ksl.nao
            nbands = ksl.bd.nbands
            mynbands = ksl.bd.mynbands
            blocksize = ksl.blocksize

            from gpaw.blacs import Redistributor
            if self.wfs.world.rank == 0:
                print('BLACS Parallelization')

            # Parallel grid descriptors
            grid = ksl.blockgrid
            assert grid.nprow * grid.npcol == self.wfs.ksl.block_comm.size
            # FOR DEBUG
            self.MM_descriptor = grid.new_descriptor(nao, nao, nao, nao)
            self.mm_block_descriptor = grid.new_descriptor(nao, nao, blocksize,
                                                           blocksize)
            self.Cnm_block_descriptor = grid.new_descriptor(nbands, nao,
                                                            blocksize,
                                                            blocksize)
            # self.CnM_descriptor = ksl.blockgrid.new_descriptor(nbands,
            #     nao, mynbands, nao)
            self.mM_column_descriptor = ksl.single_column_grid.new_descriptor(
                nao, nao, ksl.naoblocksize, nao)
            self.CnM_unique_descriptor = ksl.single_column_grid.new_descriptor(
                nbands, nao, mynbands, nao)

            # Redistributors
            self.mm2MM = Redistributor(ksl.block_comm,
                                       self.mm_block_descriptor,
                                       self.MM_descriptor)  # XXX FOR DEBUG
            self.MM2mm = Redistributor(ksl.block_comm,
                                       self.MM_descriptor,
                                       self.mm_block_descriptor)  # FOR DEBUG
            self.Cnm2nM = Redistributor(ksl.block_comm,
                                        self.Cnm_block_descriptor,
                                        self.CnM_unique_descriptor)
            self.CnM2nm = Redistributor(ksl.block_comm,
                                        self.CnM_unique_descriptor,
                                        self.Cnm_block_descriptor)
            self.mM2mm = Redistributor(ksl.block_comm,
                                       self.mM_column_descriptor,
                                       self.mm_block_descriptor)

            for kpt in self.wfs.kpt_u:
                scalapack_zero(self.mm_block_descriptor, kpt.S_MM, 'U')
                scalapack_zero(self.mm_block_descriptor, kpt.T_MM, 'U')

            # XXX to propagator class
            if self.propagator == 'taylor' and self.blacs:
                # cholS_mm = self.mm_block_descriptor.empty(dtype=complex)
                for kpt in self.wfs.kpt_u:
                    kpt.invS_MM = kpt.S_MM.copy()
                    scalapack_inverse(self.mm_block_descriptor,
                                      kpt.invS_MM, 'L')
            if self.propagator == 'taylor' and not self.blacs:
                tmp = inv(self.wfs.kpt_u[0].S_MM)
                self.wfs.kpt_u[0].invS = tmp

        # Reset the density mixer
        self.density.set_mixer(DummyMixer())
        self.tddft_initialized = True
        for k, kpt in enumerate(self.wfs.kpt_u):
            kpt.C2_nM = kpt.C_nM.copy()
            # kpt.firstC_nM = kpt.C_nM.copy()

    def update_projectors(self):
        self.timer.start('LCAO update projectors')
        # Loop over all k-points
        for k, kpt in enumerate(self.wfs.kpt_u):
            for a, P_ni in kpt.P_ani.items():
                P_ni.fill(117)
                gemm(1.0, self.wfs.P_aqMi[a][kpt.q], kpt.C_nM, 0.0, P_ni, 'n')
        self.timer.stop('LCAO update projectors')

    def get_hamiltonian(self, kpt):
        eig = self.wfs.eigensolver
        H_MM = eig.calculate_hamiltonian_matrix(self.hamiltonian, self.wfs,
                                                kpt, root=-1)
        return H_MM

    def save_wfs(self):
        for k, kpt in enumerate(self.wfs.kpt_u):
            kpt.C2_nM[:] = kpt.C_nM

    def update_hamiltonian(self):
        self.update_projectors()
        self.density.update(self.wfs)
        self.hamiltonian.update(self.density)

    def propagate_single(self, dt):
        # --------------
        # Predictor step
        # --------------
        # 1. Calculate H(t)
        self.save_wfs()  # kpt.C2_nM = kpt.C_nM
        # 2. H_MM(t) = <M|H(t)|H>
        #    Solve Psi(t+dt) from (S_MM - 0.5j*H_MM(t)*dt) Psi(t+dt) =
        #                              (S_MM + 0.5j*H_MM(t)*dt) Psi(t)

        for k, kpt in enumerate(self.wfs.kpt_u):
            if self.fxc is not None:
                if self.time == 0.0:
                    kpt.deltaXC_H_MM = self.get_hamiltonian(kpt)
                    self.hamiltonian.xc = XC(self.fxc)
                    self.update_hamiltonian()
                    assert len(self.wfs.kpt_u) == 1
                    kpt.deltaXC_H_MM -= self.get_hamiltonian(kpt)

        self.update_hamiltonian()

        # Call registered callback functions
        self.call_observers(self.niter)

        for k, kpt in enumerate(self.wfs.kpt_u):
            kpt.H0_MM = self.get_hamiltonian(kpt)
            if self.fxc is not None:
                kpt.H0_MM += kpt.deltaXC_H_MM
            self.propagate_wfs(kpt.C_nM, kpt.C_nM, kpt.S_MM, kpt.H0_MM, dt)
        # ---------------
        # Propagator step
        # ---------------
        # 1. Calculate H(t+dt)
        self.update_hamiltonian()
        # 2. Estimate H(t+0.5*dt) ~ H(t) + H(t+dT)
        for k, kpt in enumerate(self.wfs.kpt_u):
            kpt.H0_MM *= 0.5
            if self.fxc is not None:
                #  Store this to H0_MM and maybe save one extra H_MM of
                # memory?
                kpt.H0_MM += 0.5 * (self.get_hamiltonian(kpt) +
                                    kpt.deltaXC_H_MM)
            else:
                #  Store this to H0_MM and maybe save one extra H_MM of
                # memory?
                kpt.H0_MM += 0.5 * self.get_hamiltonian(kpt)

            # 3. Solve Psi(t+dt) from
            # (S_MM - 0.5j*H_MM(t+0.5*dt)*dt) Psi(t+dt)
            #    = (S_MM + 0.5j*H_MM(t+0.5*dt)*dt) Psi(t)
            self.propagate_wfs(kpt.C2_nM, kpt.C_nM, kpt.S_MM, kpt.H0_MM, dt)
        self.niter += 1
        self.time += dt

    def propagate(self, time_step=10, iterations=2000, out='lcao.dm',
                  dump_interval=50):
        assert self.wfs.dtype == complex
        assert len(self.wfs.kpt_u) == 1

        time_step *= attosec_to_autime
        self.time_step = time_step
        self.dump_interval = dump_interval
        self.tdmaxiter = self.niter + iterations

        if self.wfs.world.rank == 0:
            if self.time < self.time_step:
                self.dm_file = open(out, 'w')

                header = ('# Kick = [%22.12le, %22.12le, %22.12le]\n' %
                          (self.kick_strength[0], self.kick_strength[1],
                           self.kick_strength[2]))
                header += ('# %15s %15s %22s %22s %22s\n' %
                           ('time', 'norm', 'dmx', 'dmy', 'dmz'))
                self.dm_file.write(header)
                self.dm_file.flush()
                self.log('About to do %d propagation steps.' % iterations)
            else:
                self.dm_file = open(out, 'a')
                self.log('About to continue from iteration %d and do %d '
                         'propagation steps' % (self.niter, self.tdmaxiter))
        self.tddft_init()

        dm0 = None  # Initial dipole moment
        self.timer.start('Propagate')
        while self.niter < self.tdmaxiter:
            dm = self.density.finegd.calculate_dipole_moment(
                self.density.rhot_g)
            if dm0 is None:
                dm0 = dm

            norm = self.density.finegd.integrate(self.density.rhot_g)
            line = ('%20.8lf %20.8le %22.12le %22.12le %22.12le'
                    % (self.time, norm, dm[0], dm[1], dm[2]))
            T = localtime()
            if self.wfs.world.rank == 0:
                print(line, file=self.dm_file)

            if self.wfs.world.rank == 0 and self.niter % 1 == 0:
                self.log('iter: %3d  %02d:%02d:%02d %11.2f   %9.1f' %
                         (self.niter, T[3], T[4], T[5],
                          self.time * autime_to_attosec,
                          ln(abs(norm) + 1e-16) / ln(10)))
                self.dm_file.flush()
            self.propagate_single(self.time_step)
            
        self.call_observers(self.niter, final=True)
        if self.wfs.world.rank == 0:
            self.dm_file.close()
        self.timer.stop('Propagate')
