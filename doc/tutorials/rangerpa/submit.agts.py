def agts(queue):
    return

    si_gs = queue.add('si.groundstate.py', ncpus=1, walltime=2)
    si_rRPA = queue.add('si.range_rpa.py', deps=si_gs, ncpus=8, walltime=30)
    queue.add('si.compare.py', deps=si_rRPA, ncpus=1, walltime=1)
