import optparse
import os.path as op
import subprocess
import sys

from ase.utils import import_module

import gpaw
import gpaw.fftw as fftw
from gpaw.utilities import compiled_with_sl, compiled_with_libvdwxc


def info():
    """Show versions of GPAW and its dependencies."""
    results = [('python-' + sys.version.split()[0], sys.executable)]
    for name in ['gpaw', 'ase', 'numpy', 'scipy']:
        try:
            module = import_module(name)
        except ImportError:
            results.append((name, False))
        else:
            results.append((name + '-' + module.__version__,
                            module.__file__.rsplit('/', 1)[0] + '/'))
    module = import_module('_gpaw')
    results.append(('_gpaw',
                    op.normpath(getattr(module, '__file__', 'built-in'))))
    p = subprocess.Popen(['which', 'gpaw-python'], stdout=subprocess.PIPE)
    results.append(('parallel', p.communicate()[0].strip().decode() or False))
    results.append(('FFTW', fftw.FFTPlan is fftw.FFTWPlan))
    results.append(('scalapack', compiled_with_sl()))
    results.append(('libvdwxc', compiled_with_libvdwxc()))
    paths = ['{0}: {1}'.format(i + 1, path)
             for i, path in enumerate(gpaw.setup_paths)]
    results.append(('PAW-datasets', '\n                '.join(paths)))

    for a, b in results:
        if isinstance(b, bool):
            b = ['no', 'yes'][b]
        print('{0:16}{1}'.format(a, b))

        
def main(args):
    parser = optparse.OptionParser(usage='Usage: gpaw version',
                                   description=info.__doc__)
    parser.parse_args(args)
    info()
