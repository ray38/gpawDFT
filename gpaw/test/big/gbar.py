import os
import subprocess

from gpaw.test.big.agts import Cluster


class GbarCluster(Cluster):
       
    def submit(self, job):
        dir = os.getcwd()
        os.chdir(job.dir)
        jobdir = os.getcwd()

        self.write_pylab_wrapper(job)

#        gpaw_platform = os.environ['FYS_PLATFORM'].replace('-el6', '-2.6')

        ppn = '%d' % job.ncpus
        nodes = 1
        queueopts = '-l nodes=%d:ppn=%s' % (nodes, ppn)

        p = subprocess.Popen(
            ['qsub',
             queueopts,
             '-l',
             'walltime=%d:%02d:00' %
             (job.walltime // 3600, job.walltime % 3600 // 60),
             '-N',
             job.name,
             '-q',
             'hpc'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)

        out, err = p.communicate(
            'cd %s\n' % jobdir +
            'touch %s.start\n' % job.name +
            'mpirun gpaw-python %s.py %s > %s.output\n' %
            (job.script, job.args, job.name) +
            'echo $? > %s.done\n' % job.name)
        assert p.returncode == 0
        id = out.split('.')[0]
        job.pbsid = id
        os.chdir(dir)
