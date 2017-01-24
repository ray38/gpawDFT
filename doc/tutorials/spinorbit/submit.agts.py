def agts(queue):
    Pt_gs = queue.add('Pt_gs.py', ncpus=4, walltime=20)
    Pt_bands = queue.add('Pt_bands.py', deps=Pt_gs, ncpus=32, walltime=100)
    queue.add('plot_Pt_bands.py', ncpus=1, deps=Pt_bands,
              walltime=10, creates='Pt_bands.png')
    
    WS2_gs = queue.add('WS2_gs.py', ncpus=4, walltime=20)
    WS2_bands = queue.add('WS2_bands.py', deps=WS2_gs, ncpus=32,
                          walltime=200)
    queue.add('plot_WS2_bands.py', ncpus=1, deps=WS2_bands,
              walltime=10, creates='WS2_bands.png')
    
    Fe_gs = queue.add('Fe_gs.py', ncpus=4, walltime=20)
    Fe_bands = queue.add('Fe_bands.py', deps=Fe_gs, ncpus=32, walltime=100)
    queue.add('plot_Fe_bands.py', ncpus=1, deps=Fe_bands,
              walltime=10, creates='Fe_bands.png')

    Bi2Se3_gs = queue.add('gs_Bi2Se3.py', ncpus=4, walltime=40)
    Bi2Se3_bands = queue.add('Bi2Se3_bands.py', deps=Bi2Se3_gs, ncpus=32,
                             walltime=300)
    high_sym = queue.add('high_sym.py', deps=Bi2Se3_gs, ncpus=4,
                         walltime=30)
    queue.add('parity.py', deps=high_sym, ncpus=1, walltime=5)
    queue.add('plot_Bi2Se3_bands.py', ncpus=1, deps=Bi2Se3_bands,
              walltime=2, creates='Bi2Se3_bands.png')

    Co_gs = queue.add('gs_Co.py', ncpus=32, walltime=120)
    ani = queue.add('anisotropy.py', deps=Co_gs, ncpus=1, walltime=300)
    queue.add('plot_anisotropy.py', ncpus=1, deps=ani,
              walltime=2, creates='anisotropy.png')
