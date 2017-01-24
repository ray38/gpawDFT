from gpaw.utilities import compiled_with_libvdwxc
from gpaw.xc.libvdwxc import libvdwxc_has_pfft

def agts(queue):
    # Is walltime minutes or seconds?  If it crashes, we will know.
    if compiled_with_libvdwxc():
        queue.add('libvdwxc-example.py', ncpus=1, walltime=1)
        if libvdwxc_has_pfft():
            queue.add('libvdwxc-pfft-example.py', ncpus=8, walltime=1)
