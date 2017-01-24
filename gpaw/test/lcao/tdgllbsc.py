from ase import Atoms
from gpaw import GPAW, FermiDirac, Mixer, __version__
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.mpi import world
from gpaw.tddft import TDDFT as GRIDTDDFT
from gpaw.test import equal

ref_values = [[-0.0153712, -0.0153712, -0.0153712],
              [0.0037263, 0.0037263, 0.0037263],
              [0.0118144, 0.0118144, 0.0118144],
              [-0.0902301, -0.0902301, -0.0902301],
              [-0.0835503, -0.0835503, -0.08355028]]
calcs = []

# Cross-check fd and lcao with PBE and GLLBSC.
# Also test GLLBSC-ground state + ALDA-response in both modes.
for (mode, TDDFT) in [('lcao', LCAOTDDFT),
                      ('fd', GRIDTDDFT)]:
    for (xc, fxc) in [('PBE', 'PBE'),
                      ('GLLBSC', 'GLLBSC'),
                      ('GLLBSC', 'LDA')]:
        print('XXXXXXX', mode, xc, fxc)

        if xc == 'PBE' and mode == 'fd':
            # There are other checks for this combination
            continue

        tag = 'np%i_ver%s_%s_%s+%s' % (world.size, __version__, mode, xc, fxc)

        # silane
        atoms = Atoms('SiH4',
                      [(0.0000, 0.0000, 0.0000),
                       (0.8544, 0.8544, 0.8544),
                       (-0.8544, -0.8544, 0.8544),
                       (0.8544, -0.8544, -0.8544),
                       (-0.8544, 0.8544, -0.8544)])
        atoms.center(vacuum=4.0)

        # 1) ground state
        calcs.append(GPAW(nbands=7,
                          mixer=Mixer(0.20, 5, weight=50.0),
                          convergence={'eigenstates': 1e-6,
                                       'density': 1e-6,
                                       'energy': 1e-6},
                          h=0.40,
                          mode={'name': mode, 'force_complex_dtype': True},
                          xc=xc,
                          basis='dzp',
                          txt='grr.txt'))

        atoms.set_calculator(calcs[-1])
        energy = atoms.get_potential_energy()
        calcs[-1].write('gs.gpw', mode='all')

        # 2) restart ground state
        calcs.append(GPAW('gs.gpw', txt='grr2.txt'))
        calcs[-1].set(occupations=FermiDirac(0.05))
        calcs[-1].get_potential_energy()
        calcs[-1].write('gs.gpw', mode='all')

        # 3) time propagation
        calcs.append(TDDFT('gs.gpw', txt='grr3.txt'))
        if fxc != xc:
            calcs[-1].linearize_to_xc(fxc)

        calcs[-1].propagate(10.0, 20, 'dm.%s.dat' % tag)
        dens = calcs[-1].density
        equal(dens.finegd.calculate_dipole_moment(dens.rhot_g), 0.0, 1.0e-6)
        calcs[-1].write('td.gpw', mode='all')

        # 4) restart time propagation + apply kick
        if xc == fxc:  # TODO: restart when linearize_to_xc was applied
            calcs.append(TDDFT('td.gpw', txt='grr4.txt'))

        calcs[-1].absorption_kick([0.01, 0.01, 0.01])
        calcs[-1].propagate(10.0, 30, 'dm.%s.dat' % tag)

        dens = calcs[-1].density
        equal(dens.finegd.calculate_dipole_moment(dens.rhot_g),
              ref_values.pop(0), 1.0e-5,
              msg="Failed with %s/%s+%s: " % (mode, xc, fxc),
              fail=1)
