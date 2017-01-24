"""User provided customizations.

Here one changes the default arguments for compiling _gpaw.so (serial)
and gpaw-python (parallel).

Here are all the lists that can be modified:

* libraries
* library_dirs
* include_dirs
* extra_link_args
* extra_compile_args
* runtime_library_dirs
* extra_objects
* define_macros
* mpi_libraries
* mpi_library_dirs
* mpi_include_dirs
* mpi_runtime_library_dirs
* mpi_define_macros

To override use the form:

    libraries = ['somelib', 'otherlib']

To append use the form

    libraries += ['somelib', 'otherlib']
"""

# compiler = 'gcc'
# mpicompiler = 'mpicc'  # use None if you don't want to build a gpaw-python
# mpilinker = 'mpicc'
# platform_id = ''
# scalapack = False

# Use ScaLAPACK:
# Warning! At least scalapack 2.0.1 is required!
# See https://trac.fysik.dtu.dk/projects/gpaw/ticket/230
if scalapack:
    libraries += ['scalapack-openmpi',
                  'blacsCinit-openmpi',
                  'blacs-openmpi']
    define_macros += [('GPAW_NO_UNDERSCORE_CBLACS', '1')]
    define_macros += [('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]

# LibXC:
# In order to link libxc installed in a non-standard location
# (e.g.: configure --prefix=/home/user/libxc-2.0.1-1), use:

# - static linking:
if 0:
    include_dirs += ['/home/user/libxc-2.0.1-1/include']
    extra_link_args += ['/home/user/libxc-2.0.1-1/lib/libxc.a']
    if 'xc' in libraries:
        libraries.remove('xc')

# - dynamic linking (requires rpath or setting LD_LIBRARY_PATH at runtime):
if 0:
    include_dirs += ['/home/user/libxc-2.0.1-1/include']
    library_dirs += ['/home/user/libxc-2.0.1-1/lib']
    # You can use rpath to avoid changing LD_LIBRARY_PATH:
    # extra_link_args += ['-Wl,-rpath=/home/user/libxc-2.0.1-1/lib']
    if 'xc' not in libraries:
        libraries.append('xc')


# libvdwxc:
if 0:
    libvdwxc = True
    path = '/home/user/libvdwxc'
    extra_link_args += ['-Wl,-rpath=%s/lib' % path]
    library_dirs += ['%s/lib' % path]
    include_dirs += ['%s/include' % path]
    libraries += ['vdwxc']


# Build MPI-interface into _gpaw.so:
if 0:
    compiler = 'mpicc'
    define_macros += [('PARALLEL', '1')]
    mpicompiler = None
