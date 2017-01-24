def agts(queue):
    queue.add('gaps.py', creates='si-gaps.csv')
    eos = queue.add('eos.py', ncpus=4, walltime=600)
    queue.add('plot_a.py', deps=eos, creates='si-a.png')
