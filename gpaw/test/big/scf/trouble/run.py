import optparse
import traceback
from time import time
from glob import glob

import ase.db

import gpaw.mpi
from gpaw import GPAW


op = optparse.OptionParser(
    usage='python run.py dir [tes1.py test2.py ...]',
    description='Run tests in dir which must contain a file called ' +
    'params.py defining a calc(atoms) function that sets the eigensolver ' +
    'and other stuff')
opts, args = op.parse_args()

tag = args.pop(0).rstrip('/')
exec(open(tag + '/params.py').read())  # get the calc() function
setup = globals()['calc']

c = ase.db.connect('results.db')

if args:
    names = args
else:
    names = [name for name in glob('*.py') if name not in
             ['run.py', 'params.py',
              'submit.agts.py', 'run.py.py', 'analyse.py']]
    
for name in names:
    namespace = {}
    exec(open(name).read(), namespace)
    atoms = namespace['atoms']
    ncpus = namespace.get('ncpus', 8)
    
    if gpaw.mpi.size != ncpus:
        continue
        
    name = name[:-3]

    id = c.reserve(name=name, tag=tag)
    if not id:
        continue
        
    if atoms.calc is None:
        atoms.calc = GPAW()

    setup(atoms)
    
    atoms.calc.set(txt=tag + '/' + name + '.txt')
    
    t = time()
    try:
        e1 = atoms.get_potential_energy()
        ok = True
    except:
        ok = False
        if gpaw.mpi.rank == 0:
            traceback.print_exc(file=open(tag + '/' + name + '.error', 'w'))
    t = time() - t

    c.write(atoms, name=name, tag=tag, ok=ok,
            time=t, iters=atoms.calc.get_number_of_iterations(), ncpus=ncpus)
    
    del c[id]
