def agts(queue):
    d = [queue.add('dipole.py', ncpus=4, walltime=60),
         queue.add('pwdipole.py')]
    queue.add('plot.py', deps=d,
              creates=['zero.png', 'periodic.png', 'corrected.png',
                       'pwcorrected.png', 'slab.png'])
    queue.add('check.py', deps=d)
