def agts(queue):
    # generate = queue.add('generate.py', ncpus=1, walltime=20)
    G = [queue.add('g21gpaw.py %d' % i, walltime=40 * 60)
         for i in range(4)]
    queue.add('analyse.py', deps=G)
