from __future__ import print_function

import numpy as np

from gpaw.mpi import have_mpi
from gpaw.utilities import compiled_with_libvdwxc
from gpaw.utilities.grid_redistribute import Domains, general_redistribute
from gpaw.utilities.timing import nulltimer
from gpaw.xc.functional import XCFunctional
from gpaw.xc.gga import GGA, gga_vars, add_gradient_correction
from gpaw.xc.libxc import LibXC
from gpaw.xc.mgga import MGGA

import _gpaw


def libvdwxc_has_mpi():
    return have_mpi and _gpaw.libvdwxc_has('mpi')


def libvdwxc_has_pfft():
    return have_mpi and _gpaw.libvdwxc_has('pfft')


def check_grid_descriptor(gd):
    assert gd.parsize_c[1] == 1 and gd.parsize_c[2] == 1
    nxpts_p = gd.n_cp[0][1:] - gd.n_cp[0][:-1]
    nxpts0 = nxpts_p[0]
    for nxpts in nxpts_p[1:-1]:
        assert nxpts == nxpts0
    assert nxpts_p[-1] <= nxpts0


def get_domains(N_c, parsize_c):
    # We want a distribution like this:
    #   [B, B, ..., B, remainder, 0, 0, ..., 0].
    # with blocksize B chosen as large as possible for better load balance.
    # This function returns the blocksize and the cumulative sum of indices
    # starting with 0.
    blocksize_c = -(-N_c // parsize_c)
    return (np.arange(1 + parsize_c) * blocksize_c).clip(0, N_c)


def get_auto_pfft_grid(size):
    nproc1 = size
    nproc2 = 1
    while nproc1 > nproc2 and nproc1 % 2 == 0:
        nproc1 /= 2
        nproc2 *= 2
    return nproc1, nproc2


_VDW_NUMERICAL_CODES = {'vdW-DF': 1,
                        'vdW-DF2': 2,
                        'vdW-DF-CX': 3}


class LibVDWXC(object):
    """Minimum-tomfoolery object-oriented interface to libvdwxc."""
    def __init__(self, funcname, N_c, cell_cv, comm, mode='auto',
                 pfft_grid=None):
        self.initialized = False
        if not compiled_with_libvdwxc():
            raise ImportError('libvdwxc not compiled into GPAW')

        self.vdw_functional_name = funcname
        code = _VDW_NUMERICAL_CODES[funcname]
        self.shape = tuple(N_c)
        ptr = np.empty(1, np.intp)
        _gpaw.libvdwxc_create(ptr, code, self.shape,
                              tuple(np.ravel(cell_cv)))
        # assign ptr only now that it is initialized (so __del__ always works)
        self._ptr = ptr

        # Choose mode automatically if necessary:
        if mode == 'auto':
            if pfft_grid is not None:
                mode = 'pfft'
            elif comm.size > 1:
                mode = 'mpi'
            else:
                mode = 'serial'
        assert mode in ['serial', 'mpi', 'pfft']

        if mode != 'serial' and not have_mpi:
            raise ImportError('MPI not available for libvdwxc-%s '
                              'because GPAW is serial' % mode)

        if mode == 'pfft':
            if pfft_grid is None:
                pfft_grid = get_auto_pfft_grid(comm.size)
            nx, ny = pfft_grid
            assert nx * ny == comm.size
            # User might have passed a list, but we make sure to store a tuple:
            self.pfft_grid = (nx, ny)
        elif pfft_grid is not None:
            raise ValueError('pfft_grid specified with mode %s' % mode)

        self.mode = mode
        self.comm = comm

    def _init(self):
        assert not self.initialized

        mode = self.mode
        comm = self.comm
        if mode == 'serial':
            assert comm.size == 1, ('You cannot run in serial with %d cores'
                                    % comm.size)
            _gpaw.libvdwxc_init_serial(self._ptr)
        elif comm.get_c_object() is None:
            from gpaw.mpi import DryRunCommunicator
            # XXXXXX liable to cause really nasty errors.
            assert isinstance(comm, DryRunCommunicator)
        elif mode == 'mpi':
            if not libvdwxc_has_mpi():
                raise ImportError('libvdwxc not compiled with MPI')
            _gpaw.libvdwxc_init_mpi(self._ptr, comm.get_c_object())
        elif mode == 'pfft':
            nx, ny = self.pfft_grid
            _gpaw.libvdwxc_init_pfft(self._ptr, comm.get_c_object(), nx, ny)

        self.initialized = True

    def calculate(self, n_g, sigma_g, dedn_g, dedsigma_g):
        """Calculate energy and add partial derivatives to arrays."""
        if not self.initialized:
            self._init()

        for arr in [n_g, sigma_g, dedn_g, dedsigma_g]:
            assert arr.flags.contiguous
            assert arr.dtype == float
            # XXX We cannot actually ask libvdwxc about its expected shape
            # assert arr.shape == self.shape, [arr.shape, self.shape]
        energy = _gpaw.libvdwxc_calculate(self._ptr, n_g, sigma_g,
                                          dedn_g, dedsigma_g)
        return energy

    def get_description(self):
        if self.mode == 'serial':
            pardesc = 'fftw3 serial'
        elif self.mode == 'mpi':
            size = self.comm.size
            cores = '%d cores' % size if size != 1 else 'one core'
            pardesc = 'fftw3-mpi with %s' % cores
        else:
            assert self.mode == 'pfft'
            pardesc = 'pfft with %d x %d CPU grid' % self.pfft_grid
        return '%s [libvdwxc/%s]' % (self.vdw_functional_name, pardesc)

    def tostring(self):
        return _gpaw.libvdwxc_tostring(self._ptr)

    def __del__(self):
        if hasattr(self, '_ptr'):
            _gpaw.libvdwxc_free(self._ptr)


class RedistWrapper:
    """Call libvdwxc redistributing automatically from and to GPAW grid."""
    def __init__(self, libvdwxc, distribution, timer=nulltimer,
                 vdwcoef=1.0):
        # It is hacky for the RedistWrapper to apply the vdwcoef, but this
        # is the only accessible place where we take copies of the arrays,
        # and therefore the only 'good' place to apply a factor without
        # applying it to the existing contents of the array.
        self.libvdwxc = libvdwxc
        self.distribution = distribution
        self.timer = timer
        self.vdwcoef = vdwcoef

    def calculate(self, n_g, sigma_g, v_g, dedsigma_g):
        zeros = self.distribution.block_zeros
        nblock_g = zeros()
        sigmablock_g = zeros()
        vblock_g = zeros()
        dedsigmablock_g = zeros()

        self.timer.start('redistribute')
        self.distribution.gd2block(n_g, nblock_g)
        self.distribution.gd2block(sigma_g, sigmablock_g)
        self.timer.stop('redistribute')

        self.timer.start('libvdwxc nonlocal')
        energy = self.libvdwxc.calculate(nblock_g, sigmablock_g,
                                         vblock_g, dedsigmablock_g)
        self.timer.stop('libvdwxc nonlocal')
        energy *= self.vdwcoef
        for arr in vblock_g, dedsigmablock_g:
            arr *= self.vdwcoef

        self.timer.start('redistribute')
        self.distribution.block2gd_add(vblock_g, v_g)
        self.distribution.block2gd_add(dedsigmablock_g, dedsigma_g)
        self.timer.stop('redistribute')
        return energy


class FFTDistribution:
    def __init__(self, gd, parsize_c):
        self.input_gd = gd
        assert np.product(parsize_c) == gd.comm.size
        self.local_input_size_c = gd.n_c
        self.domains_in = Domains(gd.n_cp)
        N_c = gd.get_size_of_global_array(pad=True)
        self.domains_out = Domains([get_domains(N_c[i], parsize_c[i])
                                    for i in range(3)])

        # The auxiliary gd actually is used *only* for the rank/parpos
        # correspondence.  The actual domains it defines are unused!!
        self.aux_gd = gd.new_descriptor(comm=gd.comm, parsize_c=parsize_c)
        parpos_c = self.aux_gd.get_processor_position_from_rank()

        self.local_output_size_c = tuple(self.domains_out.get_box(parpos_c)[1])

    def block_zeros(self, shape=(),):
        return np.zeros(shape + self.local_output_size_c)

    def gd2block(self, a_xg, b_xg):
        general_redistribute(self.input_gd.comm,
                             self.domains_in, self.domains_out,
                             self.input_gd.get_processor_position_from_rank,
                             self.aux_gd.get_processor_position_from_rank,
                             a_xg, b_xg, behavior='overwrite')

    def block2gd_add(self, a_xg, b_xg):
        general_redistribute(self.input_gd.comm,
                             self.domains_out, self.domains_in,
                             self.aux_gd.get_processor_position_from_rank,
                             self.input_gd.get_processor_position_from_rank,
                             a_xg, b_xg, behavior='add')


class VDWXC(XCFunctional):
    def __init__(self, semilocal_xc, name, mode='auto',
                 pfft_grid=None, libvdwxc_name=None,
                 setup_name='revPBE', vdwcoef=1.0):
        """Initialize VDWXC object (further initialization required).

        mode can be 'auto', 'serial', 'mpi', or 'pfft'.

         * 'serial' uses FFTW and only works with serial decompositions.

         * 'mpi' uses FFTW-MPI with communicator of the grid
           descriptor, parallelizing along the x axis.

         * 'pfft' uses PFFT and works with any decomposition,
           parallelizing along two directions for best scalability.

         * 'auto' uses PFFT if pfft_grid is given, else FFTW-MPI if the
           calculation uses more than one core, else serial FFTW.

         pfft_grid is the 2D CPU grid used by PFFT and can be a tuple
         (nproc1, nproc2) that multiplies to total communicator size,
         or None.  It is an error to specify pfft_grid unless using
         PFFT.  If left unspecified, a hopefully reasonable automatic
         choice will be made.
         """
        XCFunctional.__init__(self, semilocal_xc.kernel.name,
                              semilocal_xc.kernel.type)
        # Really, 'type' should be something along the lines of vdw-df.
        self.semilocal_xc = semilocal_xc

        # We set these in the initialize later (ugly).
        self.libvdwxc = None
        self.distribution = None
        self.redist_wrapper = None
        self.timer = nulltimer
        self.setup_name = setup_name

        # These parameters are simply stored for later forwarding
        self.name = name
        if libvdwxc_name is None:
            libvdwxc_name = name
        self._libvdwxc_name = libvdwxc_name
        self._mode = mode
        self._pfft_grid = pfft_grid
        self._vdwcoef = vdwcoef

        self.last_nonlocal_energy = None
        self.last_semilocal_energy = None

        # XXXXXXXXXXXXXXXXX
        self.calculate_paw_correction = semilocal_xc.calculate_paw_correction
        #self.stress_tensor_contribution = semilocal_xc.stress_tensor_contribution
        self.calculate_spherical = semilocal_xc.calculate_spherical
        self.apply_orbital_dependent_hamiltonian = semilocal_xc.apply_orbital_dependent_hamiltonian
        self.add_forces = semilocal_xc.add_forces
        self.get_kinetic_energy_correction = semilocal_xc.get_kinetic_energy_correction
        self.rotate = semilocal_xc.rotate

    def __str__(self):
        tokens = [self._mode]
        if self._libvdwxc_name != self.name:
            tokens.append('nonlocal-name={0}'.format(self._libvdwxc_name))
            tokens.append('gga-kernel={0}'
                          .format(self.semilocal_xc.kernel.name))
        if self._pfft_grid is not None:
            tokens.append('pfft={0}'.format(self._pfft_grid))
        if self._vdwcoef != 1.0:
            tokens.append('vdwcoef={0}'.format(self._vdwcoef))

        qualifier = ', '.join(tokens)
        return '{0} [libvdwxc/{1}]'.format(self.name, qualifier)

    def set_grid_descriptor(self, gd):
        if self.gd is not None and self.gd != gd:
            raise NotImplementedError('Cannot switch grids')
        if self.libvdwxc is None:
            self._initialize(gd)
        XCFunctional.set_grid_descriptor(self, gd)
        self.semilocal_xc.set_grid_descriptor(gd)

    def get_description(self):
        lines = []
        app = lines.append
        app(self.libvdwxc.get_description())
        app('GGA kernel: %s' % self.semilocal_xc.kernel.name)
        #app('libvdwxc parameters for non-local correlation:')
        #app(self.libvdwxc.tostring())
        return '\n'.join(lines)

    def summary(self, log):
        from ase.units import Hartree
        enl = self.libvdwxc.comm.sum(self.last_nonlocal_energy)
        esl = self.gd.comm.sum(self.last_semilocal_energy)
        # In the current implementation these communicators have the same
        # processes always:
        assert self.libvdwxc.comm.size == self.gd.comm.size
        log('Non-local %s correlation energy: %.6f' % (self.name,
                                                       enl * Hartree))
        log('Semilocal %s energy: %.6f' % (self.semilocal_xc.kernel.name,
                                           esl * Hartree))
        log('(Not including atomic contributions)')

    def get_setup_name(self):
        return self.setup_name

    def _initialize(self, gd):
        N_c = gd.get_size_of_global_array(pad=True)
        self.libvdwxc = LibVDWXC(self._libvdwxc_name, N_c, gd.cell_cv,
                                 gd.comm, mode=self._mode,
                                 pfft_grid=self._pfft_grid)
        cpugrid = [1, 1, 1]
        if self.libvdwxc.mode == 'mpi':
            cpugrid[0] = gd.comm.size
        elif self.libvdwxc.mode == 'pfft':
            cpugrid[0], cpugrid[1] = self.libvdwxc.pfft_grid
        self.distribution = FFTDistribution(gd, cpugrid)
        self.redist_wrapper = RedistWrapper(self.libvdwxc,
                                            self.distribution,
                                            self.timer,
                                            self._vdwcoef)

    def set_positions(self, spos_ac):
        self.semilocal_xc.set_positions(spos_ac)

    def initialize(self, density, hamiltonian, wfs, occupations):
        self.timer = hamiltonian.timer  # fragile object robbery
        self.semilocal_xc.initialize(density, hamiltonian, wfs, occupations)
        #self.timer.start('initialize')
        #try:
        #    gd = density.xc_redistributor.aux_gd  # fragile
        #except AttributeError:
        #    gd = density.finegd
        gd = self.gd
        if wfs.world.size > gd.comm.size and np.prod(gd.N_c) > 64**3:
            # We could issue a warning if an excuse turns out to exist some day
            raise ValueError('You are using libvdwxc with only '
                             '%d out of %d available cores in a non-small '
                             'calculation (%s points).  This is not '
                             'a crime but is likely silly and therefore '
                             'triggers and error.  Please use '
                             'parallel={\'augment_grids\': True} '
                             'or complain to the developers.' %
                             (gd.comm.size, wfs.world.size,
                              ' x '.join(str(N) for N in gd.N_c)))
        #self._initialize(gd)
        # TODO Here we could decide FFT padding.
        #self.timer.stop('initialize')

    def calculate_impl(self, gd, n_sg, v_sg, e_g):
        """Calculate energy and potential.

        gd may be non-periodic.  To be distinguished from self.gd
        which is always periodic due to priminess of FFT dimensions.
        (To do: proper padded FFTs.)"""
        assert gd == self.distribution.input_gd
        assert self.libvdwxc is not None
        semiloc = self.semilocal_xc

        self.timer.start('van der Waals')

        self.timer.start('semilocal')
        # XXXXXXX taken from GGA
        grad_v = semiloc.grad_v
        sigma_xg, dedsigma_xg, gradn_svg = gga_vars(gd, grad_v, n_sg)
        n_sg[:] = np.abs(n_sg)  # XXXX What to do about this?
        sigma_xg[:] = np.abs(sigma_xg)

        # Grrr, interface still sucks
        if hasattr(semiloc, 'process_mgga'):
            semiloc.process_mgga(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg)
        else:
            semiloc.kernel.calculate(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg)
        #self.semilocal_xc.calculate_impl(gd, n_sg, v_sg, e_g)
        self.last_semilocal_energy = e_g.sum() * self.gd.dv
        self.timer.stop('semilocal')
        #energy = GGA.calculate(self, gd, n_sg, v_sg, e_g=None)

        nspins = len(n_sg)
        if nspins == 1:
            n_g = n_sg[0]
            sigma_g = sigma_xg[0]
            v_g = v_sg[0]
            dedsigma_g = dedsigma_xg[0]
        elif nspins == 2:
            n_g = n_sg.sum(0)
            sigma_g = sigma_xg[0] + 2 * sigma_xg[1] + sigma_xg[2]
            v_g = np.zeros_like(n_g)
            dedsigma_g = np.zeros_like(n_g)
        else:
            raise ValueError('Strange number of spins {0}'.format(nspins))

        energy_nonlocal = self.redist_wrapper.calculate(n_g, sigma_g,
                                                        v_g, dedsigma_g)
        if nspins == 2:
            dedsigma_xg[0] += dedsigma_g
            dedsigma_xg[1] += 2 * dedsigma_g
            dedsigma_xg[2] += dedsigma_g
            v_sg += v_g[None]

        # Note: Redistwrapper handles vdwcoef.  For now

        add_gradient_correction(grad_v, gradn_svg, sigma_xg, dedsigma_xg, v_sg)

        # XXXXXXXXXXXXXXXX ignoring vdwcoef

        # XXXXXXXXXXXXXXXX ugly
        self.last_nonlocal_energy = energy_nonlocal
        e_g[0, 0, 0] += energy_nonlocal / self.gd.dv
        self.timer.stop('van der Waals')


def vdw_df(*args, **kwargs):
    return VDWXC(semilocal_xc=GGA(LibXC('GGA_X_PBE_R+LDA_C_PW')),
                 name='vdW-DF', *args, **kwargs)


def vdw_df2(*args, **kwargs):
    return VDWXC(semilocal_xc=GGA(LibXC('GGA_X_RPW86+LDA_C_PW')),
                 name='vdW-DF2', *args, **kwargs)


def vdw_df_cx(*args, **kwargs):
    try:
        # Exists in libxc 2.2.2 or newer (or maybe from older)
        kernel = LibXC('GGA_X_LV_RPW86+LDA_C_PW')
    except NameError:
        kernel = CXGGAKernel()

    # Hidden debug feature
    if kwargs.get('gga_backend') == 'purepython':
        kernel = CXGGAKernel()
        kwargs.pop('gga_backend')
    assert 'gga_backend' not in kwargs

    return VDWXC(semilocal_xc=GGA(kernel), name='vdW-DF-CX', *args, **kwargs)


def vdw_optPBE(*args, **kwargs):
    return VDWXC(semilocal_xc=GGA(LibXC('GGA_X_OPTPBE_VDW+LDA_C_PW')),
                 name='vdW-optPBE', libvdwxc_name='vdW-DF', *args, **kwargs)


def vdw_optB88(*args, **kwargs):
    return VDWXC(semilocal_xc=GGA(LibXC('GGA_X_OPTB88_VDW+LDA_C_PW')),
                 name='optB88', libvdwxc_name='vdW-DF', *args, **kwargs)


def vdw_C09(*args, **kwargs):
    return VDWXC(semilocal_xc=GGA(LibXC('GGA_X_C09X+LDA_C_PW')),
                 name='vdW-C09', libvdwxc_name='vdW-DF', *args, **kwargs)


def vdw_beef(*args, **kwargs):
    # Kernel parameters stolen from vdw.py
    from gpaw.xc.bee import BEEVDWKernel
    kernel = BEEVDWKernel('BEE2', None,
                          0.600166476948828631066,
                          0.399833523051171368934)
    return VDWXC(semilocal_xc=GGA(kernel), name='vdW-BEEF',
                 setup_name='PBE', libvdwxc_name='vdW-DF2',
                 *args, **kwargs)


def vdw_mbeef(*args, **kwargs):
    # Note: Parameters taken from vdw.py
    from gpaw.xc.bee import BEEVDWKernel
    kernel = BEEVDWKernel('BEE3', None, 0.405258352, 0.356642240)
    return VDWXC(semilocal_xc=MGGA(kernel), name='vdW-mBEEF',
                 setup_name='PBEsol', libvdwxc_name='vdW-DF2',
                 vdwcoef=0.886774972)


# Finally, mBEEF is an MGGA.  For that we would have to un-subclass GGA
# and subclass MGGA.  Maybe the XC object architecture could be improved...


class CXGGAKernel:
    def __init__(self, just_kidding=False):
        self.just_kidding = just_kidding
        self.type = 'GGA'
        self.lda_c = LibXC('LDA_C_PW')
        if self.just_kidding:
            self.name = 'purepython rPW86_with_%s' % self.lda_c.name
        else:
            self.name = 'purepython CX'

    def calculate(self, e_g, n_sg, v_sg, sigma_xg, dedsigma_xg):
        e_g[:] = 0.0
        dedsigma_xg[:] = 0.0

        self.lda_c.calculate(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg)

        for arr in [n_sg, v_sg, sigma_xg, dedsigma_xg]:
            assert len(arr) == 1
        self._exchange(n_sg[0], sigma_xg[0], e_g, v_sg[0], dedsigma_xg[0])

    def _exchange(self, rho, grho, sx, v1x, v2x):
        """Calculate cx local exchange.

        Note that this *adds* to the energy density sx so that it can
        be called after LDA correlation part without ruining anything.
        Also it adds to v1x and v2x as is normal in GPAW."""
        tol = 1e-20
        rho[rho < tol] = tol
        grho[grho < tol] = tol
        alp = 0.021789
        beta = 1.15
        a = 1.851
        b = 17.33
        c = 0.163
        mu_LM = 0.09434
        s_prefactor = 6.18733545256027
        Ax = -0.738558766382022  # = -3./4. * (3./pi)**(1./3)
        four_thirds = 4. / 3.

        grad_rho = np.sqrt(grho)

        # eventually we need s to power 12.  Clip to avoid overflow
        # (We have individual tolerances on both rho and grho, but
        # they are not sufficient to guarantee this)
        s_1 = (grad_rho / (s_prefactor * rho**four_thirds)).clip(0.0, 1e20)
        s_2 = s_1 * s_1
        s_3 = s_2 * s_1
        s_4 = s_3 * s_1
        s_5 = s_4 * s_1
        s_6 = s_5 * s_1

        fs_rPW86 = (1.0 + a * s_2 + b * s_4 + c * s_6)**(1. / 15.)

        if self.just_kidding:
            fs = fs_rPW86
        else:
            fs = (1.0 + mu_LM * s_2) / (1.0 + alp * s_6) \
                + alp * s_6 / (beta + alp * s_6) * fs_rPW86

        # the energy density for the exchange.
        sx[:] += Ax * rho**four_thirds * fs

        df_rPW86_ds = (1. / (15. * fs_rPW86**14.0)) * \
            (2 * a * s_1 + 4 * b * s_3 + 6 * c * s_5)

        if self.just_kidding:
            df_ds = df_rPW86_ds  # XXXXXXXXXXXXXXXXXXXX
        else:
            df_ds = 1. / (1. + alp * s_6)**2 \
                * (2.0 * mu_LM * s_1 * (1. + alp * s_6) -
                   6.0 * alp * s_5 * (1. + mu_LM * s_2)) \
                + alp * s_6 / (beta + alp * s_6) * df_rPW86_ds \
                + 6.0 * alp * s_5 * fs_rPW86 / (beta + alp * s_6) \
                * (1. - alp * s_6 / (beta + alp * s_6))

        # de/dn.  This is the partial derivative of sx wrt. n, for s constant
        v1x[:] += Ax * four_thirds * (rho**(1. / 3.) * fs -
                                      grad_rho / (s_prefactor * rho) * df_ds)
        # de/d(nabla n).  The other partial derivative
        v2x[:] += 0.5 * Ax * df_ds / (s_prefactor * grad_rho)
        # (We may or may not understand what that grad_rho is doing here.)


def test_derivatives():
    gen = np.random.RandomState(1)
    shape = (1, 20, 20, 20)
    ngpts = np.product(shape)
    n_sg = gen.rand(*shape)
    sigma_xg = np.zeros(shape)
    sigma_xg[:] = gen.rand(*shape)

    qe_kernel = CXGGAKernel(just_kidding=True)
    libxc_kernel = LibXC('GGA_X_RPW86+LDA_C_PW')

    cx_kernel = CXGGAKernel(just_kidding=False)

    def check(kernel, n_sg, sigma_xg):
        e_g = np.zeros(shape[1:])
        dedn_sg = np.zeros(shape)
        dedsigma_xg = np.zeros(shape)
        kernel.calculate(e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg)
        return e_g, dedn_sg, dedsigma_xg

    def check_and_write(kernel):
        n1_sg = n_sg.copy()
        e_g, dedn_sg, dedsigma_xg = check(kernel, n_sg, sigma_xg)
        dedn = dedn_sg[0, 0, 0, 0]
        dedsigma = dedsigma_xg[0, 0, 0, 0]

        dn = 1e-6
        n1_sg = n_sg.copy()
        n1_sg[0, 0, 0, 0] -= dn / 2.
        e1_g, _, _ = check(kernel, n1_sg, sigma_xg)

        n1_sg[0, 0, 0, 0] += dn
        e2_g, _, _ = check(kernel, n1_sg, sigma_xg)

        dedn_fd = (e2_g[0, 0, 0] - e1_g[0, 0, 0]) / dn
        dedn_err = abs(dedn - dedn_fd)

        print('e', e_g.sum() / ngpts)
        print('dedn', dedn, 'fd', dedn_fd, 'err %e' % dedn_err)

        sigma1_xg = sigma_xg.copy()
        sigma1_xg[0, 0, 0, 0] -= dn / 2.
        e1s_g, _, _ = check(kernel, n_sg, sigma1_xg)

        sigma1_xg[0, 0, 0, 0] += dn
        e2s_g, _, _ = check(kernel, n_sg, sigma1_xg)

        dedsigma_fd = (e2s_g[0, 0, 0] - e1s_g[0, 0, 0]) / dn
        dedsigma_err = dedsigma - dedsigma_fd

        print('dedsigma', dedsigma, 'fd', dedsigma_fd, 'err %e' % dedsigma_err)
        return e_g, dedn_sg, dedsigma_xg

    print('pw86r libxc')
    e_lxc_g, dedn_lxc_g, dedsigma_lxc_g = check_and_write(libxc_kernel)
    print()
    print('pw86r ours')
    e_qe_g, dedn_qe_g, dedsigma_qe_g = check_and_write(qe_kernel)
    print()
    print('cx')
    check_and_write(cx_kernel)

    print()
    print('lxc vs qe discrepancies')
    print('=======================')
    e_err = np.abs(e_lxc_g - e_qe_g).max()
    print('e', e_err)
    dedn_err = np.abs(dedn_qe_g - dedn_lxc_g).max()
    dedsigma_err = np.abs(dedsigma_lxc_g - dedsigma_qe_g).max()
    print('dedn', dedn_err)
    print('dedsigma', dedsigma_err)


def test_selfconsistent():
    from gpaw import GPAW
    from ase.build import molecule
    from gpaw.xc.gga import GGA

    system = molecule('H2O')
    system.center(vacuum=3.)

    def test(xc):
        calc = GPAW(mode='lcao',
                    xc=xc,
                    setups='sg15',
                    txt='gpaw.%s.txt' % str(xc)  # .kernel.name
                    )
        system.set_calculator(calc)
        return system.get_potential_energy()

    libxc_results = {}

    for name in ['GGA_X_PBE_R+LDA_C_PW', 'GGA_X_RPW86+LDA_C_PW']:
        xc = GGA(LibXC(name))
        e = test(xc)
        libxc_results[name] = e

    cx_gga_results = {}
    cx_gga_results['rpw86'] = test(GGA(CXGGAKernel(just_kidding=True)))
    cx_gga_results['lv_rpw86'] = test(GGA(CXGGAKernel(just_kidding=False)))

    vdw_results = {}
    vdw_coef0_results = {}

    for vdw in [vdw_df(), vdw_df2(), vdw_df_cx()]:
        vdw.vdwcoef = 0.0
        vdw_coef0_results[vdw.__class__.__name__] = test(vdw)
        vdw.vdwcoef = 1.0  # Leave nicest text file by running real calc last
        vdw_results[vdw.__class__.__name__] = test(vdw)

    from gpaw.mpi import world
    # These tests basically verify that the LDA/GGA parts of vdwdf
    # work correctly.
    if world.rank == 0:
        print('Now comparing...')
        err1 = cx_gga_results['rpw86'] - libxc_results['GGA_X_RPW86+LDA_C_PW']
        print('Our rpw86 must be identical to that of libxc. Err=%e' % err1)
        print('RPW86 interpolated with Langreth-Vosko stuff differs by %f'
              % (cx_gga_results['lv_rpw86'] - cx_gga_results['rpw86']))
        print('Each vdwdf with vdwcoef zero must yield same result as gga'
              'kernel')
        err_df1 = vdw_coef0_results['VDWDF'] - libxc_results['GGA_X_PBE_R+'
                                                             'LDA_C_PW']
        print('  df1 err=%e' % err_df1)
        err_df2 = vdw_coef0_results['VDWDF2'] - libxc_results['GGA_X_RPW86+'
                                                              'LDA_C_PW']
        print('  df2 err=%e' % err_df2)
        err_cx = vdw_coef0_results['VDWDFCX'] - cx_gga_results['lv_rpw86']
        print('   cx err=%e' % err_cx)


if __name__ == '__main__':
    test_derivatives()
    # test_selfconsistent()
