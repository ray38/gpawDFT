from __future__ import print_function
import numpy as np
import ase.db

con = ase.db.connect('g2_dzp.db')
N = 6  # number of optimizers
M = len(con) // N  # number of molecules
data = []
for row in con.select():
    data.append((row.optimizer,
                 row.name,
                 row.get('time', 42),
                 row.get('steps', 42),
                 row.get('energy', 42),
                 row.get('fmax', 42)))
data.sort()
optimizers = [x[0] for x in data[::M]]
data = np.array([x[2:] for x in data]).reshape((N, M, 4))
e0 = data[:, :, 2].min(0)
results = []
for opt, d in zip(optimizers, data):
    ok = (d[:, 3] < 0.05) & (d[:, 2] < e0 + 0.1)
    failed = M - sum(ok)
    time, niter = d[ok, :2].mean(0)
    results.append((failed, time, niter, opt))

with open('g2_dzp.csv', 'w') as f:
    for failed, time, niter, opt in sorted(results):
        print('{0}, {1}, {2:.1f}, {3:.1f}'.format(opt, failed, time, niter),
              file=f)
