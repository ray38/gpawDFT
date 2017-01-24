from __future__ import print_function, division
import multiprocessing as mp
import glob
import os
import pickle
import random
import re
import time
import traceback

import numpy as np
from ase import Atoms
from ase.data import covalent_radii, atomic_numbers, chemical_symbols
from ase.build import bulk
from ase.build import fcc111
from ase.units import Bohr

from gpaw import GPAW, PW, setup_paths, Mixer, ConvergenceError
from gpaw.atom.generator2 import _generate, DatasetGenerationError


my_covalent_radii = covalent_radii.copy()
my_covalent_radii[1] += 0.2
my_covalent_radii[2] += 0.2
my_covalent_radii[4] -= 0.1  # Be
my_covalent_radii[5] -= 0.1
my_covalent_radii[6] -= 0.1
my_covalent_radii[7] -= 0.1
my_covalent_radii[8] -= 0.1
my_covalent_radii[9] -= 0.1  # F
my_covalent_radii[10] += 0.2
for e in ['Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']:  # missing radii
    my_covalent_radii[atomic_numbers[e]] = 1.7


NN = 11


class GA:
    def __init__(self, initialvalue=None):
        self.initialvalue = initialvalue

        self.individuals = {}
        self.errors = {}

        if os.path.isfile('pool.csv'):
            for line in open('pool.csv'):
                words = line.split(',')
                n = int(words.pop(0))
                error = float(words.pop(0))
                x = tuple(int(word) for word in words[:-NN])
                self.individuals[x] = (error, n)
                y = tuple(float(word) for word in words[-NN:])
                self.errors[n] = y

        self.fd = open('pool.csv', 'a')  # pool of genes
        self.n = len(self.individuals)

    def run(self, func, sleep=5, mutate=5.0, size1=2, size2=1000):
        pool = mp.Pool()  # process pool
        results = []
        while True:
            while len(results) < mp.cpu_count():
                x = self.new(mutate, size1, size2)
                self.individuals[x] = (np.inf, self.n)
                result = pool.apply_async(func, [self.n, x])
                self.n += 1
                results.append(result)
            time.sleep(sleep)
            for result in results:
                if result.ready():
                    break
            else:
                continue
            results.remove(result)
            n, x, errors, error = result.get()
            self.individuals[x] = (error, n)
            print('{0},{1},{2},{3}'.format(n, error,
                                           ','.join(str(i) for i in x),
                                           ','.join('{0:.4f}'.format(e)
                                                    for e in errors)),
                  file=self.fd)
            self.fd.flush()

            best = sorted(self.individuals.values())[:20]
            nbest = [N for e, N in best]
            for f in glob.glob('[0-9]*.txt'):
                if int(f[:-4]) not in nbest:
                    os.remove(f)

            if len(self.individuals) > 400 and best[0][0] == np.inf:
                for result in results:
                    result.wait()
                return

    def new(self, mutate, size1, size2):
        if len(self.individuals) == 0:
            return self.initialvalue

        if len(self.individuals) < 32:
            x3 = np.array(self.initialvalue, dtype=float)
            mutate *= 2.5
        else:
            all = sorted((y, x)
                         for x, y in self.individuals.items()
                         if y is not None)  # and y != np.inf)
            S = len(all)
            if S < size1:
                x3 = np.array(self.initialvalue, dtype=float)
            else:
                parents = random.sample(all[:size1], 2)
                if S > size1 and random.random() < 0.33:
                    i = random.randint(size1, min(S, size2) - 1)
                    parents.append(all[i])
                    del parents[random.randint(0, 1)]
                else:
                    mutate /= 3

                x1 = parents[0][1]
                x2 = parents[1][1]
                r = np.random.rand(len(x1))
                x3 = r * x1 + (1 - r) * x2

        while True:
            x3 += np.random.normal(0, mutate, len(x3))
            x = tuple(int(round(a)) for a in x3)
            if x not in self.individuals:
                break

        return x


def read_reference(name, symbol):
    for line in open('../../{0}.csv'.format(name)):
        words = line.split(',')
        if words[0] == 'c':
            x = [float(word) for word in words[2:]]
        elif words[0] == symbol:
            ref = dict((c, float(word))
                       for c, word in zip(x, words[2:]))
            ref['a'] = float(words[1])
            return ref


def fit(E):
    em, e0, ep = E
    a = (ep + em) / 2 - e0
    b = (ep - em) / 2
    return -b / (2 * a)


class DatasetOptimizer:
    tolerances = np.array([0.3,
                           0.01, 0.03, 0.05,
                           0.01, 0.03, 0.05,
                           40,
                           400,  # 0.1 eV convergence
                           0.0005,  # eggbox error
                           0.4])

    conf = None

    def __init__(self, symbol='H', nc=False):
        self.old = False

        self.symbol = symbol
        self.nc = nc

        with open('../start.txt') as fd:
            for line in fd:
                words = line.split()
                if words[1] == symbol:
                    projectors = words[3]
                    radii = [float(f) for f in words[5].split(',')]
                    r0 = float(words[7].split(',')[1])
                    break
            else:
                raise ValueError

        # Parse projectors string:
        pattern = r'(-?\d+\.\d)'
        energies = []
        for m in re.finditer(pattern, projectors):
            energies.append(float(projectors[m.start():m.end()]))
        self.projectors = re.sub(pattern, '%.1f', projectors)
        self.nenergies = len(energies)

        # Round to integers:
        x = ([e / 0.1 for e in energies] +
             [r / 0.05 for r in radii] +
             [r0 / 0.05])
        self.x = tuple(int(round(f)) for f in x)

        # Read FHI-Aims data:
        self.reference = {'fcc': read_reference('fcc', symbol),
                          'rocksalt': read_reference('rocksalt', symbol)}

        self.ecut1 = 450.0
        self.ecut2 = 800.0

        setup_paths[:0] = ['../..', '.']

        self.Z = atomic_numbers[symbol]
        self.rc = my_covalent_radii[self.Z]
        self.rco = my_covalent_radii[8]

    def minimize(self):
        fd = open('pool.csv', 'w')
        done = {}

        def f(x):
            n, x, errors, error = self(0, tuple(x))
            n = len(done)
            done[tuple(x)] = error
            print('{0},{1},{2},{3}'.format(n, error,
                                           ','.join(str(i) for i in x),
                                           ','.join('{0:.4f}'.format(e)
                                                    for e in errors)),
                  file=fd)
            fd.flush()
            return error

        x = list(self.x)

        e0 = f(x)
 
        while True:
            ee = []
            for d in [1, -1]:
                for i, yi in enumerate(x):
                    x[i] += d
                    if tuple(x) in done:
                        e = done[tuple(x)]
                    else:
                        e = f(x)
                    x[i] -= d
                    ee.append(e)

            i = np.argmin(ee)
            if ee[i] > e0:
                break
            e0 = ee[i]
            x[i % len(x)] += 1 - (i // len(x)) * 2

    def run(self):  # , mu, n1, n2):
        # mu = float(mu)
        # n1 = int(n1)
        # n2 = int(n2)
        ga = GA(self.x)
        ga.run(self)  # , mutate=mu, size1=n1, size2=n2)

    def run_initial(self):
        errors, total_error = self(0, self.x)[2:]
        print(self.symbol, 'Errors:', errors, '\nTotal error:', total_error)
        with open(self.symbol + '.pckl', 'wb') as f:
            pickle.dump((self.symbol, errors, total_error), f)

    def best(self, N=None):
        ga = GA(self.x)
        best = sorted((error, id, x)
                      for x, (error, id) in ga.individuals.items())
        if 0:
            import pickle
            pickle.dump(sorted(ga.individuals.values()), open('Zn.pckl', 'wb'))
        if N is None:
            return len(best), best[0] + (ga.errors[best[0][1]],)
        else:
            return [(error, id, x, ga.errors[id])
                    for error, id, x in best[:N]]

    def summary(self, N=10):
        # print('dFffRrrICEer:')
        for error, id, x, errors in self.best(N):
            params = [0.1 * p for p in x[:self.nenergies]]
            params += [0.05 * p for p in x[self.nenergies:]]
            print('{0:2} {1:2}{2:4}{3:6.1f}|{4}|'
                  '{5:4.1f}|'
                  '{6:6.2f} {7:6.2f} {8:6.2f}|'
                  '{9:6.2f} {10:6.2f} {11:6.2f}|'
                  '{12:3.0f} {13:4.0f} {14:7.4f} {15:4.1f}'
                  .format(self.Z,
                          self.symbol,
                          id, error,
                          ' '.join('{0:5.2f}'.format(p) for p in params),
                          *errors))

    def best1(self):
        try:
            n, (error, id, x, errors) = self.best()
        except IndexError:
            return
        energies, radii, r0, projectors = self.parameters(x)
        if 0:
            print(error, self.symbol, n)
        if 1:
            if projectors[-1].isupper():
                nderiv0 = 5
            else:
                nderiv0 = 2
            fmt = '{0:2} {1:2} -P {2:31} -r {3:20} -0 {4},{5:.2f} # {6:10.3f}'
            print(fmt.format(self.Z,
                             self.symbol,
                             projectors,
                             ','.join('{0:.2f}'.format(r) for r in radii),
                             nderiv0,
                             r0,
                             error))
        if 0:
            with open('parameters.txt', 'w') as fd:
                print(projectors, ' '.join('{0:.2f}'.format(r)
                                           for r in radii + [r0]),
                      file=fd)
        if 0 and error != np.inf and error != np.nan:
            self.generate(None, 'PBE', projectors, radii, r0, True, '',
                          logderivs=not False)

    def generate(self, fd, xc, projectors, radii, r0,
                 scalar_relativistic=False, tag=None, logderivs=True):
        if self.old:
            self.generate_old(fd, xc, scalar_relativistic, tag)
            return 0.0

        if projectors[-1].isupper():
            nderiv0 = 5
        else:
            nderiv0 = 2

        type = 'poly'
        if self.nc:
            type = 'nc'
        gen = _generate(self.symbol, xc, self.conf, projectors, radii,
                        scalar_relativistic, None, r0, nderiv0,
                        (type, 4), None, None, fd)
        if not scalar_relativistic:
            if not gen.check_all():
                print('dataset check failed')
                return np.inf

        if tag is not None:
            gen.make_paw_setup(tag or None).write_xml()

        r = 1.1 * gen.rcmax

        lmax = 2
        if 'f' in projectors:
            lmax = 3

        error = 0.0
        if logderivs:
            for l in range(lmax + 1):
                emin = -1.5
                emax = 2.0
                n0 = gen.number_of_core_states(l)
                if n0 > 0:
                    e0_n = gen.aea.channels[l].e_n
                    emin = max(emin, e0_n[n0 - 1] + 0.1)
                energies = np.linspace(emin, emax, 100)
                de = energies[1] - energies[0]
                ld1 = gen.aea.logarithmic_derivative(l, energies, r)
                ld2 = gen.logarithmic_derivative(l, energies, r)
                error += abs(ld1 - ld2).sum() * de

        return error

    def generate_old(self, fd, xc, scalar_relativistic, tag):
        from gpaw.atom.configurations import parameters
        from gpaw.atom.generator import Generator
        par = parameters[self.symbol]
        g = Generator(self.symbol, xc, scalarrel=scalar_relativistic,
                      nofiles=True, txt=fd)
        g.run(exx=True, logderiv=False, use_restart_file=False, name=tag,
              **par)

    def parameters(self, x):
        energies = tuple(0.1 * i for i in x[:self.nenergies])
        radii = [0.05 * i for i in x[self.nenergies:-1]]
        r0 = 0.05 * x[-1]
        projectors = self.projectors % energies
        return energies, radii, r0, projectors

    def __call__(self, n, x):
        fd = open('{0}.txt'.format(n), 'w')
        energies, radii, r0, projectors = self.parameters(x)

        if r0 < 0.2:
            print(n, x, 'core radius too small')
            return n, x, [np.inf] * NN, np.inf

        if any(r < r0 for r in radii):  # or any(e <= 0.0 for e in energies):
            print(n, x, 'radii too small')
            return n, x, [np.inf] * NN, np.inf

        errors = self.test(n, fd, projectors, radii, r0)

        try:
            os.remove('{0}.ga{1}.PBE'.format(self.symbol, n))
        except OSError:
            pass

        return n, x, errors, ((errors / self.tolerances)**2).sum()

    def test(self, n, fd, projectors, radii, r0):
        error = self.generate(fd, 'PBE', projectors, radii, r0,
                              tag='ga{0}'.format(n))
        for xc in ['PBE', 'LDA', 'PBEsol', 'RPBE', 'PW91']:
            error += self.generate(fd, xc, projectors, radii, r0,
                                   scalar_relativistic=True)

        if not np.isfinite(error):
            return [np.inf] * NN

        results = {'dataset': error}

        for name in ['slab', 'fcc', 'rocksalt', 'convergence', 'eggbox']:
            try:
                result = getattr(self, name)(n, fd)
            except ConvergenceError:
                print(n, name)
                result = np.inf
            except Exception as ex:
                print(n, name, ex)
                traceback.print_exc()
                result = np.inf
            results[name] = result

        rc = self.rc / Bohr
        results['radii'] = sum(r - rc for r in radii if r > rc)

        errors = self.calculate_total_error(fd, results)

        return errors

    def calculate_total_error(self, fd, results):
        errors = [results['dataset']]
        maxiter = results['slab']

        for name in ['fcc', 'rocksalt']:
            result = results[name]
            if isinstance(result, dict):
                maxiter = max(maxiter, result['maxiter'])
                errors.append(result['a'] - result['a0'])
                errors.append(result['c90'] - result['c90ref'])
                errors.append(result['c80'] - result['c80ref'])
            else:
                maxiter = np.inf
                errors.extend([np.inf, np.inf, np.inf])

        errors.append(maxiter)
        errors.append(results['convergence'])

        errors.append(results['eggbox'])

        errors.append(results['radii'])

        return errors

    def fcc(self, n, fd):
        ref = self.reference['fcc']
        a0r = ref['a']  # scalar-relativistic minimum
        sc = min(0.8, 2 * self.rc * 2**0.5 / a0r)
        sc = min((abs(s - sc), s) for s in ref if s != 'a')[1]
        maxiter = 0
        energies = []
        M = 200
        mixer = {}
        if 58 <= self.Z <= 70 or 90 <= self.Z <= 102:
            M = 999
            mixer = {'mixer': Mixer(0.01, 5)}
        for s in [sc, 0.9, 0.95, 1.0, 1.05]:
            atoms = bulk(self.symbol, 'fcc', a0r * s)
            atoms.calc = GPAW(mode=PW(self.ecut2),
                              kpts={'density': 4.0, 'even': True},
                              xc='PBE',
                              setups='ga' + str(n),
                              maxiter=M,
                              txt=fd,
                              **mixer)
            e = atoms.get_potential_energy()
            maxiter = max(maxiter, atoms.calc.get_number_of_iterations())
            energies.append(e)

        return {'c90': energies[1] - energies[3],
                'c80': energies[0] - energies[3],
                'c90ref': ref[0.9] - ref[1.0],
                'c80ref': ref[sc] - ref[1.0],
                'a0': fit([ref[s] for s in [0.95, 1.0, 1.05]]) * 0.05 * a0r,
                'a': fit(energies[2:]) * 0.05 * a0r,
                'maxiter': maxiter}

    def rocksalt(self, n, fd):
        ref = self.reference['rocksalt']
        a0r = ref['a']
        sc = min(0.8, 2 * (self.rc + self.rco) / a0r)
        sc = min((abs(s - sc), s) for s in ref if s != 'a')[1]
        maxiter = 0
        energies = []
        M = 200
        mixer = {}
        if 58 <= self.Z <= 70 or 90 <= self.Z <= 102:
            M = 999
            mixer = {'mixer': Mixer(0.01, 5)}
        for s in [sc, 0.9, 0.95, 1.0, 1.05]:
            atoms = bulk(self.symbol + 'O', 'rocksalt', a0r * s)
            atoms.calc = GPAW(mode=PW(self.ecut2),
                              kpts={'density': 4.0, 'even': True},
                              xc='PBE',
                              setups={self.symbol: 'ga' + str(n)},
                              maxiter=M,
                              txt=fd,
                              **mixer)
            e = atoms.get_potential_energy()
            maxiter = max(maxiter, atoms.calc.get_number_of_iterations())
            energies.append(e)

        return {'c90': energies[1] - energies[3],
                'c80': energies[0] - energies[3],
                'c90ref': ref[0.9] - ref[1.0],
                'c80ref': ref[sc] - ref[1.0],
                'a0': fit([ref[s] for s in [0.95, 1.0, 1.05]]) * 0.05 * a0r,
                'a': fit(energies[2:]) * 0.05 * a0r,
                'maxiter': maxiter}

    def slab(self, n, fd):
        a0 = self.reference['fcc']['a']
        atoms = fcc111(self.symbol, (1, 1, 7), a0, vacuum=3.5)
        assert not atoms.pbc[2]
        M = 333
        if 58 <= self.Z <= 70 or 90 <= self.Z <= 102:
            M = 1333
            mixer = {'mixer': Mixer(0.001, 3, 100)}
        else:
            mixer = {}
        atoms.calc = GPAW(mode=PW(self.ecut1),
                          kpts={'density': 2.0, 'even': True},
                          xc='PBE',
                          setups='ga' + str(n),
                          maxiter=M,
                          txt=fd,
                          **mixer)
        atoms.get_potential_energy()
        itrs = atoms.calc.get_number_of_iterations()
        return itrs

    def eggbox(self, n, fd):
        h = 0.18
        a0 = 16 * h
        atoms = Atoms(self.symbol, cell=(a0, a0, 2 * a0), pbc=True)
        M = 333
        if 58 <= self.Z <= 70 or 90 <= self.Z <= 102:
            M = 999
            mixer = {'mixer': Mixer(0.01, 5)}
        else:
            mixer = {}
        atoms.calc = GPAW(h=h,
                          xc='PBE',
                          symmetry='off',
                          setups='ga' + str(n),
                          maxiter=M,
                          txt=fd,
                          **mixer)
        atoms.positions += h / 2  # start with broken symmetry
        e0 = atoms.get_potential_energy()
        atoms.positions -= h / 6
        e1 = atoms.get_potential_energy()
        atoms.positions -= h / 6
        e2 = atoms.get_potential_energy()
        atoms.positions -= h / 6
        e3 = atoms.get_potential_energy()
        return np.ptp([e0, e1, e2, e3])

    def convergence(self, n, fd):
        a = 3.0
        atoms = Atoms(self.symbol, cell=(a, a, a), pbc=True)
        M = 333
        if 58 <= self.Z <= 70 or 90 <= self.Z <= 102:
            M = 999
            mixer = {'mixer': Mixer(0.01, 5)}
        else:
            mixer = {}
        atoms.calc = GPAW(mode=PW(1500),
                          xc='PBE',
                          setups='ga' + str(n),
                          symmetry='off',
                          maxiter=M,
                          txt=fd,
                          **mixer)
        e0 = atoms.get_potential_energy()
        de0 = 0.0
        for ec in range(800, 200, -100):
            atoms.calc.set(mode=PW(ec))
            atoms.calc.set(eigensolver='rmm-diis')
            de = abs(atoms.get_potential_energy() - e0)
            if de > 0.1:
                ec0 = ec + (de - 0.1) / (de - de0) * 100
                return ec0
            de0 = de
        return 250


if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser(usage='python -m gpaw.atom.optimize '
                                   '[options] element',
                                   description='Optimize dataset')
    parser.add_option('-s', '--summary', action='store_true')
    parser.add_option('-b', '--best', action='store_true')
    parser.add_option('-r', '--run', action='store_true')
    parser.add_option('-m', '--minimize', action='store_true')
    parser.add_option('-n', '--norm-conserving', action='store_true')
    parser.add_option('-i', '--initial-only', action='store_true')
    parser.add_option('-o', '--old-setups', action='store_true')
    opts, args = parser.parse_args()
    if opts.run or opts.minimize:
        symbol = args[0]
        if not os.path.isdir(symbol):
            os.mkdir(symbol)
        os.chdir(symbol)
        do = DatasetOptimizer(symbol, opts.norm_conserving)
        if opts.run:
            if opts.initial_only:
                do.old = opts.old_setups
                do.run_initial()
            else:
                do.run()
        else:
            do.minimize()
    else:
        if args == ['.']:
            symbol = os.getcwd().rsplit('/', 1)[1]
            args = [symbol]
            os.chdir('..')
        elif len(args) == 0:
            args = [symbol for symbol in chemical_symbols
                    if os.path.isdir(symbol)]
        for symbol in args:
            os.chdir(symbol)
            try:
                do = DatasetOptimizer(symbol, opts.norm_conserving)
            except ValueError:
                pass
            else:
                if opts.summary:
                    do.summary(15)
                elif opts.best:
                    do.best1()
            os.chdir('..')
