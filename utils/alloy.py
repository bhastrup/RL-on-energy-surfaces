
# https://wiki.fysik.dtu.dk/ase/tutorials/neb/diffusion.html

from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms, FixedPlane
from asap3 import EMT
from ase.optimize import QuasiNewton, BFGS
from ase.visualize import view

from ase import Atoms
from ase.io import Trajectory
from ase.io import write
# 2x2-Al(001) surface with 3 layers and an
# Au atom adsorbed in a hollow site:
slab = fcc100('Ag', size=(3,3,3))
add_adsorbate(slab, 'Au', 1.7, 'ontop')
slab.center(axis=2, vacuum=4.0)

# Make sure the structure is correct:
# view(slab)

# Fix second and third layers:
# mask = [atom.tag > 1 for atom in slab]
# print(mask)
# slab.set_constraint(FixAtoms(mask=mask))

# Use EMT potential:
slab.calc = EMT()

# Move to hollow
slab[-1].x += slab.get_cell()[0, 0] / 6
slab[-1].y += slab.get_cell()[1, 1] / 6

# Initial state:
qn = QuasiNewton(slab, trajectory='slab.traj')
qn.run(fmax=0.05)
#view(slab)


# Final state:
slab_b = slab.copy()
slab_b.calc = EMT()
slab_b[-1].x += slab_b.get_cell()[0, 0] / 3
slab_b[-1].y += slab_b.get_cell()[1, 1] / 3

qn = QuasiNewton(slab_b, trajectory='slab_B.traj')
qn.run(fmax=0.05)
#view(slab_b)


# Specify agent atom
n_atoms = len(slab)
agent_atom = n_atoms - 1

# Calculated if agent atom is in goal hollow site
hollow_neighbors = [12, 13, 14]
dist_A = slab.get_distances(agent_atom, hollow_neighbors, mic=False)
dist_B= slab_b.get_distances(agent_atom, hollow_neighbors, mic=False)
dist_B_periodic = slab_b.get_distances(agent_atom, hollow_neighbors, mic=True)
