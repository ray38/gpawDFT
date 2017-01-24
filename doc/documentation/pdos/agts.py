def agts(queue):
    top = queue.add('top.py', ncpus=8)
    queue.add('pdos.py', deps=top, creates='pdos.png')

    calc = queue.add('lcaodos_gs.py', ncpus=8)
    queue.add('lcaodos_plt.py', deps=calc, creates='lcaodos.png')
