def agts(queue):
    gs_si = queue.add('gs_Si.py', ncpus=4, walltime=20)
    bse_si = queue.add('eps_Si.py', ncpus=4, walltime=240, deps=gs_si)
    queue.add('plot_Si.py', ncpus=1, walltime=10, deps=bse_si,
              creates='bse_Si.png')

    gs_mos2 = queue.add('gs_MoS2.py', ncpus=4, walltime=100)
    bse_mos2 = queue.add('pol_MoS2.py', ncpus=64, walltime=2000, deps=gs_mos2)
    queue.add('plot_MoS2.py', ncpus=1, walltime=10, deps=bse_mos2,
              creates='bse_MoS2.png')

    eps = queue.add('get_2d_eps.py', ncpus=1, walltime=500, deps=gs_mos2)
    queue.add('plot_2d_eps.py', ncpus=1, walltime=10, deps=eps,
              creates='2d_eps.png')

    queue.add('alpha_MoS2.py', ncpus=1, walltime=10, deps=gs_mos2)
