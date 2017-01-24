def agts(queue):
    return
    run = queue.add('pbe_gpaw_pw.py Al', ncpus=4,
                    queueopts='-l nodes=1:ppn=4:opteron4', walltime=60)
    if 0:  # run when new setups ready
        analyse = queue.add('analyse.py dcdft_pbe_gpaw_pw',
                            ncpus=1, walltime=10, deps=run,
                            creates=['dcdft_pbe_gpaw_pw.csv',
                                     'dcdft_pbe_gpaw_pw.txt',
                                     'dcdft_pbe_gpaw_pw_Delta.txt',
                                     'dcdft_pbe_gpaw_pw_raw.csv'])
        verify = queue.add('pbe_gpaw_pw_verify.py',
                           ncpus=1, walltime=10, deps=[analyse])
