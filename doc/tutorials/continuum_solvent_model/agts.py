def agts(queue):
    h2o = queue.add('ethanol_in_water.py', ncpus=4, walltime=10)
    queue.add('check.py', deps=h2o)
