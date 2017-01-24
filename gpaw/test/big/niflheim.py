import os
import subprocess

from gpaw.test.big.agts import Cluster


class NiflheimCluster(Cluster):
    def __init__(self, asepath='', setuppath='$GPAW_SETUP_PATH'):
        self.asepath = asepath
        self.setuppath = setuppath

    def submit(self, job):
        dir = os.getcwd()
        os.chdir(job.dir)
        self.write_pylab_wrapper(job)

        cmd = ['sbatch',
               '--job-name={}'.format(job.name),
               '--time={}'.format(job.walltime // 60),
               '--ntasks={}'.format(job.ncpus)]

        script = [
            '#!/bin/sh',
            'touch {}.start'.format(job.name),
            'mpirun -x OMP_NUM_THREADS=1 gpaw-python {}.py {} > {}.output'
            .format(job.script, job.args, job.name),
            'echo $? > {}.done'.format(job.name)]
        if 1:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            p.communicate(('\n'.join(script) + '\n').encode())
            assert p.returncode == 0
        os.chdir(dir)
