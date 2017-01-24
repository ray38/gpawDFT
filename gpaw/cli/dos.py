from __future__ import print_function
import sys

from ase.dft.dos import DOS

from gpaw import GPAW


def dos(filename, plot=False, output='dos.csv', width=0.1):
    """Calculate density of states.
    
    filename: str
        Name of restart-file.
    plot: bool
        Show a plot.
    output: str
        Name of CSV output file.
    width: float
        Width of Gaussians.
    """
    calc = GPAW(filename, txt=None)
    dos = DOS(calc, width)
    D = [dos.get_dos(spin) for spin in range(calc.get_number_of_spins())]
    if output:
        fd = sys.stdout if output == '-' else open(output, 'w')
        for x in zip(dos.energies, *D):
            print(*x, sep=', ', file=fd)
        if output != '-':
            fd.close()
    if plot:
        import matplotlib.pyplot as plt
        for y in D:
            plt.plot(dos.energies, y)
        plt.show()
