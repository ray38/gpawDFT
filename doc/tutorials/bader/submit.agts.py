def agts(queue):
    h2o = queue.add('h2o.py')
    bader = queue.add('bader.py', deps=h2o, creates=['ACF.dat'])
    queue.add('plot.py', deps=bader, creates=['h2o-bader.png'])
