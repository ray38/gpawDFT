import numpy as np

from gpaw.utilities.lapack import general_diagonalize
from gpaw.utilities import unpack
from gpaw.hs_operators import reshape
from gpaw.eigensolvers.eigensolver import Eigensolver


class Davidson(Eigensolver):
    """Simple Davidson eigensolver

    It is expected that the trial wave functions are orthonormal
    and the integrals of projector functions and wave functions
    ``nucleus.P_uni`` are already calculated.

    Solution steps are:

    * Subspace diagonalization
    * Calculate all residuals
    * Add preconditioned residuals to the subspace and diagonalize
    """

    def __init__(self, niter=1, smin=None, normalize=True):
        Eigensolver.__init__(self)
        self.niter = niter
        self.smin = smin
        self.normalize = normalize

        if smin is not None:
            raise NotImplementedError(
                'See https://trac.fysik.dtu.dk/projects/gpaw/ticket/248')

        self.orthonormalization_required = False

    def __repr__(self):
        return 'Davidson(niter=%d, smin=%r, normalize=%r)' % (
            self.niter, self.smin, self.normalize)

    def todict(self):
        return {'name': 'dav', 'niter': self.niter}
        
    def initialize(self, wfs):

        Eigensolver.initialize(self, wfs)
        self.overlap = wfs.overlap

        # Allocate arrays
        self.H_2n2n = np.empty((2 * self.nbands, 2 * self.nbands), self.dtype)
        self.S_2n2n = np.empty((2 * self.nbands, 2 * self.nbands), self.dtype)
        self.eps_2n = np.empty(2 * self.nbands)

    def estimate_memory(self, mem, wfs):
        Eigensolver.estimate_memory(self, mem, wfs)
        nbands = wfs.bd.nbands
        mem.subnode('H_nn', nbands * nbands * mem.itemsize[wfs.dtype])
        mem.subnode('S_nn', nbands * nbands * mem.itemsize[wfs.dtype])
        mem.subnode('H_2n2n', 4 * nbands * nbands * mem.itemsize[wfs.dtype])
        mem.subnode('S_2n2n', 4 * nbands * nbands * mem.itemsize[wfs.dtype])
        mem.subnode('eps_2n', 2 * nbands * mem.floatsize)

    def iterate_one_k_point(self, hamiltonian, wfs, kpt):
        """Do Davidson iterations for the kpoint"""
        niter = self.niter
        nbands = self.nbands
        mynbands = self.mynbands

        gd = wfs.matrixoperator.gd
        bd = self.operator.bd

        psit_nG, Htpsit_nG = self.subspace_diagonalize(hamiltonian, wfs, kpt)
        # Note that psit_nG is now in self.operator.work1_nG and
        # Htpsit_nG is in kpt.psit_nG!

        H_2n2n = self.H_2n2n
        S_2n2n = self.S_2n2n
        eps_2n = self.eps_2n

        self.timer.start('Davidson')

        if self.keep_htpsit:
            R_nG = Htpsit_nG
            psit2_nG = reshape(self.Htpsit_nG, psit_nG.shape)
        else:
            R_nG = wfs.empty(mynbands, q=kpt.q)
            psit2_nG = wfs.empty(mynbands, q=kpt.q)
            wfs.apply_pseudo_hamiltonian(kpt, hamiltonian, psit_nG, R_nG)
            wfs.pt.integrate(psit_nG, kpt.P_ani, kpt.q)
        
        self.calculate_residuals(kpt, wfs, hamiltonian, psit_nG,
                                 kpt.P_ani, kpt.eps_n, R_nG)

        def integrate(a_G, b_G):
            return np.real(wfs.integrate(a_G, b_G, global_integral=False))

        # Note on band parallelization
        # The "large" H_2n2n and S_2n2n matrices are at the moment
        # global and replicated over band communicator, and the
        # general diagonalization is performed in serial i.e. without
        # scalapack

        for nit in range(niter):
            H_2n2n[:] = 0.0
            S_2n2n[:] = 0.0

            norm_n = np.zeros(mynbands)
            error = 0.0
            for n in range(mynbands):
                if kpt.f_n is None:
                    weight = kpt.weight
                else:
                    weight = kpt.f_n[n]
                if self.nbands_converge != 'occupied':
                    if n < self.nbands_converge:
                        weight = kpt.weight
                    else:
                        weight = 0.0
                error += weight * integrate(R_nG[n], R_nG[n])

                ekin = self.preconditioner.calculate_kinetic_energy(
                    psit_nG[n:n + 1], kpt)
                psit2_nG[n] = self.preconditioner(R_nG[n:n + 1], kpt, ekin)

                if self.normalize:
                    norm_n[n] = integrate(psit2_nG[n], psit2_nG[n])

                N = bd.global_index(n)
                H_2n2n[N, N] = kpt.eps_n[n]
                S_2n2n[N, N] = 1.0

            bd.comm.sum(H_2n2n)
            bd.comm.sum(S_2n2n)

            if self.normalize:
                gd.comm.sum(norm_n)
                for norm, psit2_G in zip(norm_n, psit2_nG):
                    psit2_G *= norm**-0.5
        
            # Calculate projections
            P2_ani = wfs.pt.dict(mynbands)
            wfs.pt.integrate(psit2_nG, P2_ani, kpt.q)

            self.timer.start('calc. matrices')
            
            # Hamiltonian matrix
            # <psi2 | H | psi>

            def H(psit_xG):
                result_xG = R_nG
                wfs.apply_pseudo_hamiltonian(kpt, hamiltonian, psit_xG,
                                             result_xG)
                return result_xG

            def dH(a, P_ni):
                return np.dot(P_ni, unpack(hamiltonian.dH_asp[a][kpt.s]))

            H_nn = self.operator.calculate_matrix_elements(psit_nG, kpt.P_ani,
                                                           H, dH, psit2_nG,
                                                           P2_ani)

            H_2n2n[nbands:, :nbands] = H_nn

            # <psi2 | H | psi2>

            def H(psit_xG):
                # H | psi2 > already calculated in previous step
                result_xG = R_nG
                return result_xG

            def dH(a, P_ni):
                return np.dot(P_ni, unpack(hamiltonian.dH_asp[a][kpt.s]))

            H_nn = self.operator.calculate_matrix_elements(psit2_nG, P2_ani,
                                                           H, dH)

            H_2n2n[nbands:, nbands:] = H_nn

            # Overlap matrix
            # <psi2 | S | psi>

            def S(psit_G):
                return psit_G
            
            def dS(a, P_ni):
                return np.dot(P_ni, wfs.setups[a].dO_ii)

            S_nn = self.operator.calculate_matrix_elements(psit_nG, kpt.P_ani,
                                                           S, dS, psit2_nG,
                                                           P2_ani)

            S_2n2n[nbands:, :nbands] = S_nn

            # <psi2 | S | psi2>
            S_nn = self.operator.calculate_matrix_elements(psit2_nG, P2_ani,
                                                           S, dS)
            S_2n2n[nbands:, nbands:] = S_nn

            self.timer.stop('calc. matrices')

            self.timer.start('diagonalize')
            if gd.comm.rank == 0 and bd.comm.rank == 0:
                m = 0
                if self.smin:
                    s_N, U_NN = np.linalg.eigh(S_2n2n)
                    m = int((s_N < self.smin).sum())

                if m == 0:
                    general_diagonalize(H_2n2n, eps_2n, S_2n2n)
                else:
                    T_Nn = np.dot(U_NN[:, m:], np.diag(s_N[m:]**-0.5))
                    H_2n2n[:nbands, nbands:] = \
                        H_2n2n[nbands:, :nbands].conj().T
                    eps_2n[:-m], P_nn = np.linalg.eigh(
                        np.dot(np.dot(T_Nn.T.conj(), H_2n2n), T_Nn))
                    H_2n2n[:-m] = np.dot(T_Nn, P_nn).T

            gd.comm.broadcast(H_2n2n, 0)
            gd.comm.broadcast(eps_2n, 0)
            bd.comm.broadcast(H_2n2n, 0)
            bd.comm.broadcast(eps_2n, 0)

            self.operator.bd.distribute(eps_2n[:nbands], kpt.eps_n[:])

            self.timer.stop('diagonalize')

            self.timer.start('rotate_psi')
            # Rotate psit_nG

            # Memory references during rotate:
            # Case 1, no band parallelization:
            #   Before 1. matrix multiply: psit_nG -> operator.work1_xG
            #   After  1. matrix multiply: psit_nG -> R_nG
            #   After  2. matrix multiply: tmp_nG -> work1_xG
            #
            # Case 2, band parallelization
            # Work arrays used only in send/recv buffers,
            # psit_nG -> psit_nG
            # tmp_nG -> psit2_nG

            psit_nG = self.operator.matrix_multiply(H_2n2n[:nbands, :nbands],
                                                    psit_nG, kpt.P_ani,
                                                    out_nG=R_nG)

            tmp_nG = self.operator.matrix_multiply(H_2n2n[:nbands, nbands:],
                                                   psit2_nG, P2_ani)

            if bd.comm.size > 1:
                psit_nG += tmp_nG
            else:
                tmp_nG += psit_nG
                psit_nG, R_nG = tmp_nG, psit_nG
            
            for a, P_ni in kpt.P_ani.items():
                P2_ni = P2_ani[a]
                P_ni += P2_ni

            self.timer.stop('rotate_psi')

            if nit < niter - 1:
                wfs.apply_pseudo_hamiltonian(kpt, hamiltonian, psit_nG,
                                             R_nG)
                self.calculate_residuals(kpt, wfs, hamiltonian, psit_nG,
                                         kpt.P_ani, kpt.eps_n, R_nG)

        self.timer.stop('Davidson')
        error = gd.comm.sum(error)
        return error, psit_nG
