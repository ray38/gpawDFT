compiler = 'cc'
mpicompiler = 'cc'
mpilinker = 'cc'
scalapack=True
hdf5=True
libxc='/usr/common/software/libxc/2.1.3'
include_dirs += [libxc+'/include']
extra_link_args += [libxc+'/lib/libxc.a']
extra_compile_args += ['-O2']
if 'xc' in libraries:
    libraries.remove('xc')
# these are in the cray wrapper
if 'blas' in libraries:
    libraries.remove('blas')
if 'lapack' in libraries:
    libraries.remove('lapack')

if scalapack:
    define_macros += [('GPAW_NO_UNDERSCORE_CBLACS', '1')]
    define_macros += [('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]
