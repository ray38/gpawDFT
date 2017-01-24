from __future__ import print_function
import numpy as np
from gpaw.grid_descriptor import GridDescriptor
from gpaw.xc.libvdwxc import vdw_df, vdw_df2, vdw_df_cx, \
    vdw_optPBE, vdw_optB88, vdw_C09, vdw_beef, \
    libvdwxc_has_mpi, libvdwxc_has_pfft

# This test verifies that the results returned by the van der Waals
# functionals implemented in libvdwxc do not change.

N_c = np.array([23, 10, 6])
gd = GridDescriptor(N_c, N_c * 0.2, pbc_c=(1, 0, 1))

n_sg = gd.zeros(1)
nG_sg = gd.collect(n_sg)
if gd.comm.rank == 0:
    gen = np.random.RandomState(0)
    nG_sg[:] = gen.rand(*nG_sg.shape)
gd.distribute(nG_sg, n_sg)

for mode in ['serial', 'mpi', 'pfft']:
    if mode == 'serial' and gd.comm.size > 1:
        continue
    if mode == 'mpi' and not libvdwxc_has_mpi():
        continue
    if mode == 'pfft' and not libvdwxc_has_pfft():
        continue

    def test(vdwxcclass, Eref=np.nan, nvref=np.nan):
        xc = vdwxcclass(mode=mode)
        xc._initialize(gd)
        if gd.comm.rank == 0:
            print(xc.libvdwxc.tostring())
        v_sg = gd.zeros(1)
        E = xc.calculate(gd, n_sg, v_sg)
        nv = gd.integrate(n_sg * v_sg, global_integral=True)
        nv = float(nv)  # Comes out as an array due to spin axis

        Eerr = None if Eref is None else abs(E - Eref)
        nverr = None if nvref is None else abs(nv - nvref)

        if gd.comm.rank == 0:
            name = xc.name
            print(name)
            print('=' * len(name))
            print('E  = %19.16f vs ref = %19.16f :: err = %10.6e'
                  % (E, Eref, Eerr))
            print('nv = %19.16f vs ref = %19.16f :: err = %10.6e'
                  % (nv, nvref, nverr))
            print()
        gd.comm.barrier()

        if Eerr is not None:
            assert Eerr < 1e-14, 'error=%s' % Eerr
        if nverr is not None:
            assert nverr < 1e-14, 'error=%s' % nverr

    from functools import partial
    vdw_df_cx_part = partial(vdw_df_cx, gga_backend='purepython')

    test(vdw_df, -3.7373236650435593, -4.7766302688360334)
    test(vdw_df2, -3.75680663471042, -4.7914451465590480)
    test(vdw_df_cx_part, -3.6297336577106862, -4.6753445074468276)

    test(vdw_optPBE, -3.6836013391734239, -4.7290002029944613)
    test(vdw_optB88, -3.7182162512875037, -4.7586582439197587)
    test(vdw_C09, -3.6178542441863657, -4.6660960269614167)
    test(vdw_beef, -3.7742682115984687, -4.8520774634041866)
