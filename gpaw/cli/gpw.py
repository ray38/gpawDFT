def gpw(filename):
    """Write summary of GPAW-restart file.
    
    filename: str
        Name of restart-file.
    """
    from gpaw import GPAW
    GPAW(filename)
