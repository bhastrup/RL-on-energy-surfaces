
from ase import Atoms
from ase.io import Trajectory
from ase.io import write
from ase.visualize import view
from ase.build import fcc111, add_adsorbate

import imp
try:
    imp.find_module('asap3')
    found_asap3 = True
except ImportError:
    found_asap3 = False

if found_asap3:
    from asap3 import EMT
else:
    from ase.calculators.emt import EMT

from ase.optimize import BFGS
from ase.constraints import FixAtoms, FixedPlane

# Build ASE slab
slab = fcc111('Cu', size=(2,3,4), vacuum=10.0)
add_adsorbate(slab, 'Cu', 2.08, 'fcc')
calc = EMT()
slab.set_calculator(calc)

# Specify agent atom
n_atoms = len(slab)
agent_atom = n_atoms - 1

# Create constraint for bottom layer atoms in slabs
#active_dist = 1.5
#all_dists = slab.get_distances(
#    a=agent_atom,
#    indices=list(range(n_atoms)),
#    mic=True
#)
#mask = all_dists > active_dist
#constraint = FixAtoms(mask=mask)
#slab.set_constraint(constraint)


# Specify initial configuration, A
dyn = BFGS(slab, trajectory='slab.traj')
dyn.run(fmax=0.03)

# Specify goal configuration, B
slab_b = slab.copy()
slab_b.set_calculator(calc)
slab_b[-1].x += slab.get_cell()[0, 0] / 2
dyn_B = BFGS(slab_b, trajectory='slab_B.traj')
dyn_B.run(fmax=0.03)


# Calculated if agent atom is in goal hollow site
hollow_neighbors = [12, 13, 14]
dist_A = slab.get_distances(agent_atom, hollow_neighbors, mic=False)
dist_B= slab_b.get_distances(agent_atom, hollow_neighbors, mic=False)
dist_B_periodic = slab_b.get_distances(agent_atom, hollow_neighbors, mic=True)
