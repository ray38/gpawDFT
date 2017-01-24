def agts(queue):
    return
    a = queue.add('dcdft_gpaw.py', ncpus=4, walltime=40)
    queue.add('testdb.py', deps=a)
    queue.add('extract.py dcdft.db', deps=a, creates='dcdft.db_raw.txt')
