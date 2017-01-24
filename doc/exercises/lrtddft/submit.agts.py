def agts(queue):
    calc1 = queue.add('Na2TDDFT.py',
                      ncpus=2,
                      walltime=60)
    queue.add('part2.py', deps=calc1)
    gs = queue.add('ground_state.py', ncpus=8)
    queue.add('spectrum.py', deps=gs)
