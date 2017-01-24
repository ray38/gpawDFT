def agts(queue):
    jobs = [queue.add('g2_dzp.py ' + str(i), ncpus=4, walltime=800)
            for i in range(10)]
    queue.add('g2_dzp_csv.py', deps=jobs, creates='g2_dzp.csv')
