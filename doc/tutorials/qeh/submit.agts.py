def agts(queue):
    gs_MoS2 = queue.add('gs_MoS2.py', ncpus=16, walltime=25)
    gs_WSe2 = queue.add('gs_WSe2.py', ncpus=16, walltime=25)

    bb_MoS2 = queue.add('bb_MoS2.py', deps=gs_MoS2, ncpus=16,
                        walltime=800)
    bb_WSe2 = queue.add('bb_WSe2.py', deps=gs_WSe2, ncpus=16,
                        walltime=800)

    queue.add('interlayer.py', deps=[bb_MoS2, bb_WSe2],
              creates='W_r.svg')
