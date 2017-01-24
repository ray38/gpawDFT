def agts(queue):
    basis = queue.add('lcaotddft_basis.py', ncpus=1, walltime=10)
    ag55 = queue.add('lcaotddft_ag55.py', deps=[basis], ncpus=64, walltime=120)
    queue.add('lcaotddft_fig1.py', deps=[ag55], creates='fig1.png')
    induced = queue.add('lcaotddft_induced.py', ncpus=4, walltime=60)
    queue.add('lcaotddft_analyse.py', deps=[induced], ncpus=1, walltime=15,
              creates='Na8_imag.png')
