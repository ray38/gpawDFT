scalapack = True
libraries = ['readline',
             'gfortran',
             'scalapack',
             'openblas',
             'xc']
extra_compile_args = ['-O2', '-std=c99', '-fPIC', '-Wall']
define_macros += [('GPAW_NO_UNDERSCORE_CBLACS', '1'),
                  ('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]
platform_id = 'el7'
