"""GPAW command-line tool."""
from __future__ import print_function
import os
import sys
import inspect
import optparse
import textwrap


functions = {'xc': 'gpaw.xc.xc',
             'run': 'gpaw.cli.run.main',
             'dos': 'gpaw.cli.dos.dos',
             'rpa': 'gpaw.xc.rpa.rpa',
             'gpw': 'gpaw.cli.gpw.gpw',
             'info': 'gpaw.cli.info.main',
             'test': 'gpaw.test.test.main',
             'atom': 'gpaw.atom.aeatom.main',
             'diag': 'gpaw.fulldiag.fulldiag',
             'quick': 'gpaw.cli.quick.quick',
             'dataset': 'gpaw.atom.generator2.main',
             'symmetry': 'gpaw.symmetry.analyze_atoms',
             'install-data': 'gpaw.cli.install_data.main'}


def main():
    commands = sorted(functions.keys())
    parser1 = optparse.OptionParser(
        usage='Usage: gpaw [-h] [-v] [-P N] command [more options]\n' +
        '       gpaw command --help  (for help on individual commands)',
        description='Run one of these commands: {0}.'
        .format(', '.join(commands)))
    parser1.disable_interspersed_args()
    add = parser1.add_option
    add('-v', '--verbose', action='store_true')
    add('-P', '--parallel', type=int, metavar='N', default=1,
        help="Run on N CPUs.")
    opts1, args1 = parser1.parse_args()

    if opts1.parallel > 1:
        from gpaw.mpi import size
        if size == 1:
            # Start again using gpaw-python in parallel:
            args = ['mpiexec', '-np', str(opts1.parallel),
                    'gpaw-python']
            if args1[0] == 'python':
                args += args1[1:]
            else:
                args += sys.argv
            os.execvp('mpiexec', args)
    
    if len(args1) == 0:
        parser1.print_help()
        raise SystemExit
    command = args1[0]
    try:
        modulename, funcname = functions[command].rsplit('.', 1)
    except KeyError:
        parser1.error('Unknown command: ' + command)
    module = __import__(modulename, globals(), locals(), [funcname])
    func = getattr(module, funcname)
    kwargs = {}
    if funcname == 'main':
        args = [args1[1:]]
    else:
        nargs, optnames, parser = construct_parser(func, command)
    
        opts, args = parser.parse_args(args1[1:])
        if len(args) != nargs:
            parser.error('Wrong number of arguments!')
        for optname in optnames:
            value = getattr(opts, optname)
            if value is not None:
                kwargs[optname] = value
    try:
        func(*args, **kwargs)
    except Exception as x:
        if opts1.verbose:
            raise
        else:
            error = ('{0}: {1}'.format(x.__class__.__name__, x) +
                     '\nTo get a full traceback, use: gpaw --verbose')
            parser1.error(error)
            
            
def construct_parser(func, name):
    """Construct parser from function arguments and docstring."""
    args, varargs, keywords, defaults = inspect.getargspec(func)
    assert varargs is None
    assert keywords is None
    if defaults:
        optnames = args[-len(defaults):]
        defaults = dict(zip(optnames, defaults))
        del args[-len(defaults):]
    else:
        optnames = []
        defaults = {}
    doc = func.__doc__
    headline, doc = doc.split('\n', 1)
    lines = textwrap.dedent(doc).splitlines()
    description = None
    options = []
    shortoptions = set()
    i = 0
    while i < len(lines):
        words = lines[i].split(':')
        if (len(words) > 1 and i + 1 < len(lines) and
            lines[i + 1].startswith('    ')):
            if description is None:
                description = ' '.join([headline] + lines[:i])
            arg = words[0]
            help = []
            type = words[1].strip()
            i += 1
            while i < len(lines) and lines[i].startswith('    '):
                help.append(lines[i][4:])
                i += 1
            kwargs = {'help': ' '.join(help)}
            if arg not in defaults:
                continue
            default = defaults.get(arg)
            if type == 'bool':
                kwargs['action'] = 'store_' + str(not default).lower()
            else:
                kwargs['type'] = type
                if default is not None:
                    kwargs['default'] = default
            short = arg[0]
            if short in shortoptions:
                short = arg[1]
            shortoptions.add(short)
            options.append(optparse.Option('-' + short,
                                           '--' + arg.replace('_', '-'),
                                           **kwargs))
        else:
            if description:
                break
            i += 1
        
    epilog = ' '.join(lines[i:])
            
    parser = optparse.OptionParser(
        usage='Usage: gpaw {0} <{1}> [options]'.format(name, '> <'.join(args)),
        description=description,
        option_list=options,
        epilog=epilog)
    return len(args), optnames, parser
