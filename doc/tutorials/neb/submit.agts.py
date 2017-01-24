def agts(queue):
    d = queue.add('diffusion1.py', ncpus=4, walltime=10)
    n = queue.add('neb.py', deps=d, ncpus=12, walltime=60)
    queue.add('check.py', deps=n)

