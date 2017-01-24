import glob
import os
import subprocess

from ase.utils.build_web_page import build, git_pull, main


def build_gpaw_web_page(force_build):
    changes = git_pull('gpaw')
    if force_build or changes:
        os.chdir('ase')
        subprocess.check_call('python setup.py install --home=..', shell=True)
        os.chdir('..')
        subprocess.check_call(
            'wget --no-check-certificate --quiet --timestamping '
            'http://wiki.fysik.dtu.dk/gpaw-files/'
            'gpaw-setups-latest.tar.gz', shell=True)
        subprocess.check_call(
            'tar xf gpaw-setups-latest.tar.gz', shell=True)
        path = os.path.abspath(glob.glob('gpaw-setups-[0-9]*')[0])
        build(True, 'gpaw', 'GPAW_SETUP_PATH=' + path)
        
        
if __name__ == '__main__':
    main(build_gpaw_web_page)
