
from ase import io
from ase.neb import NEB
from ase.optimize import MDMin
from ase.calculators.emt import EMT
from gpaw import GPAW, PW, FermiDirac
from utils.slab_params import *

import numpy as np
import matplotlib.pyplot as plt

import os, sys

# Want to create new NEB optimization trajectory for each image individually?
split_traj = False

# Create directory to save analysis
script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'runs/neb/')
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

# Read initial and final states:
initial = io.read('slab.traj')
final = io.read('slab_B.traj')

# Number of images
n_im = 5

# Make a band consisting of n_im-2 intermediate images:
images = [initial]
images += [initial.copy() for i in range(n_im-2)]
images += [final]
neb = NEB(images)

# Interpolate linearly the potisions of the three middle images:
neb.interpolate()

# Set calculators:
for image in images[1:(n_im-1)]:
    # image.calc = EMT()
    image.calc = GPAW(
        mode=PW(),
        xc='PBE',
        hund=True,
        eigensolver='rmm-diis',  # This solver can parallelize over bands
        occupations=FermiDirac(0.0, fixmagmom=True),
        txt='H.out'
    ) 

# Optimize and save NEB optimization as traj
optimizer = MDMin(neb, trajectory='A2B.traj')
optimizer.run(fmax=0.04)


# Split up NEB traj file into image constituents
if split_traj == True:
    neb_path = Trajectory("A2B.traj", 'r')
    neb_split = [[neb_path[k] for k in np.arange(i, len(neb_path), n_im)] for i in range(n_im)]
    for i in range(n_im):
        neb_traj = Trajectory("neb_traj_image_" + str(i) + ".traj", 'w')
        for k in range(len(neb_split[i])):
            neb_traj.write(neb_split[i][k])


# To make the barrier plot more spatial, the barrier is plotted as a function of the 
# reaction coordinate given by the projection onto the line connection A and B.

# Calculate reaction coordinate for each image
rc = np.zeros(n_im)
rc[-1] = 1
rc[1:-1] = [np.dot(
    neb.images[i].get_positions()[agent_atom] - neb.images[0].get_positions()[agent_atom],
    neb.images[-1].get_positions()[agent_atom] - neb.images[0].get_positions()[agent_atom]) /
    np.linalg.norm(neb.images[-1].get_positions()[agent_atom] - neb.images[0].get_positions()[agent_atom])**2
    for i in range(1, n_im-1)
]


# Plot styling
fs = 12
fig_width = 10
fig_height = 12
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 50})
fig, ax = plt.subplots(
    nrows=2, ncols=1, figsize=(fig_width, fig_height), constrained_layout=True
)

# Energy barrier
neb_energies = [neb.images[i].get_potential_energy() for i in range(n_im)]
ax[0].plot(rc, neb_energies, linewidth=2)
ax[0].set(xlabel="Reaction coordinate")
ax[0].set(ylabel="Energy [eV]")

# Plot spatial height of jump
neb_height = [neb.images[i].get_positions()[agent_atom][-1] - neb.images[0].get_positions()[agent_atom][-1]
    for i in range(n_im)]
ax[1].plot(rc, neb_height, linewidth=2)
ax[1].set(xlabel="Reaction coordinate")
ax[1].set(ylabel="Height of agent atom [Å]")

fig.savefig(results_dir + 'NEB_barrier_plot.png', bbox_inches='tight')
