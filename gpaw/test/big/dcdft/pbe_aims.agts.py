# Note: due to how agts works external executables (abinit, aims, etc.)
# must be run on the submission platform (bmaster1), currently opteron4
def agts(queue):
    return
    queue.add('pbe_aims.py Al', ncpus=1,
              queueopts='-l nodes=1:ppn=4:opteron4', walltime=3*40)
