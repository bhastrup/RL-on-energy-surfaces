
from ase import Atoms
from ase.io import Trajectory
from ase.io import write

from ase.visualize import view # run view(atom_objects[0])
from ase.build import fcc111, add_adsorbate
from ase.calculators.abinit import Abinit
from ase.calculators.emt import EMT
# from asap3 import EMT
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS
from ase.constraints import FixAtoms

# Build ASE slab
slab = fcc111('Cu', size=(2,3,4), vacuum=10.0)
add_adsorbate(slab, 'Cu', 2.08, 'fcc')
calc = EMT()
#calc = LennardJones()
slab.set_calculator(calc)
#view(slab)

# Specify initial configuration, A
dyn = BFGS(slab, trajectory='slab.traj')
dyn.run(fmax=0.05)


# Specify goal configuration, B
slab_b = slab.copy()
slab_b[-1].x += slab.get_cell()[0, 0] / 2
dyn_B = BFGS(slab, trajectory='slab_B.traj')
dyn_B.run(fmax=0.05)
#view(slab_b)

# Specify agent atom
n_atoms = len(slab)
agent_atom = n_atoms - 1

# Calculated if agent atom is in goal hollow site
hollow_neighbors = [12, 13, 14]
dist_A = slab.get_distances(agent_atom, hollow_neighbors, mic=False)
dist_B= slab_b.get_distances(agent_atom, hollow_neighbors, mic=False)
dist_B_periodic = slab_b.get_distances(agent_atom, hollow_neighbors, mic=True)

