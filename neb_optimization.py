
# NEB code: https://gitlab.com/ase/ase/-/blob/f153e5aeef80bda03284de9740967c47ba81232b/ase/neb.py

from ase import io
from ase.neb import NEB
from ase.optimize import MDMin
# from gpaw import GPAW, PW, FermiDirac
# from utils.slab_params import *
from Al_alloy import *

import numpy as np
import matplotlib.pyplot as plt

import os, sys
from datetime import datetime
now = datetime.now()
dt_string = now.strftime("%d_%m_%H_%M")

# Create directory to save analysis
script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'runs/neb/')
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

# Want to create new NEB optimization trajectory for each image individually?
split_traj_image = False
split_traj_band = True
dft = False

# Constraints, https://wiki.fysik.dtu.dk/ase/_modules/ase/constraints.html
constrain_bottom = True
active_dist = 7
constrain_agent_y = False

# Number of images
n_im = 11

# NEB spring konstant
k_spring = 0.1


# Read initial and final states:
initial = io.read('slab.traj')
final = io.read('slab_B.traj')

# Make a band consisting of n_im-2 intermediate images:
images = [initial]
images += [initial.copy() for i in range(n_im-2)]
images += [final]
neb = NEB(images, k=k_spring)

# Create constraint for bottom layer atoms in slabs
if constrain_bottom:
    
    all_dists = initial.get_distances(
        a=agent_atom,
        indices=list(range(n_atoms)),
        mic=True
    )
    mask = all_dists > active_dist
    bottom_constraint = FixAtoms(mask=mask)

# Set calculators and constrain images:
for image in images[1:(n_im-1)]:
    if dft:
        image.calc = GPAW(
            mode=PW(),
            xc='PBE',
            hund=True,
            eigensolver='rmm-diis',  # This solver can parallelize over bands
            occupations=FermiDirac(0.0, fixmagmom=True),
            txt='H.out'
        )
    else:
        image.calc = EMT()
    if constrain_bottom:
        image.set_constraint(image.constraints + [bottom_constraint])
    if constrain_agent_y:
        image.set_constraint(
            image.constraints + [FixedPlane(agent_atom, np.array([0, 1, 0]))]
        )

# Interpolate linearly the potisions of the three middle images:
neb.interpolate()

# Break symmetry
for i in range(1, (n_im-1)):
    neb.images[i][agent_atom].x += 0.5
    neb.images[i][agent_atom].y += -0.5

# Optimize and save NEB optimization as traj
# optimizer = MDMin(neb, trajectory='A2B.traj')
optimizer = MDMin(neb, trajectory=results_dir + 'A2B.traj')
optimizer.run(fmax=0.05)


# Split up NEB traj file into image constituents
if split_traj_image == True:
    neb_path = Trajectory(results_dir + "A2B.traj", 'r')
    neb_split = [[neb_path[k] for k in np.arange(i, len(neb_path), n_im)] for i in range(n_im)]
    for i in range(n_im):
        neb_traj = Trajectory("neb_traj_image_" + str(i) + ".traj", 'w')
        for k in range(len(neb_split[i])):
            neb_traj.write(neb_split[i][k])

if split_traj_band == True:
    neb_path = Trajectory(results_dir + "A2B.traj", 'r')
    neb_split = [neb_path[k:k+n_im] for k in np.arange(0, len(neb_path)-n_im, n_im)]
    neb_traj_band = Trajectory(results_dir + "neb_traj_band_" + dt_string + ".traj", 'w')
    for band in neb_split:
        for atom_object in band:
            neb_traj_band.write(atom_object)



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
fig_width = 8
fig_height = 8
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 10})
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

fig.savefig(results_dir + 'NEB_barrier_plot_' + dt_string + '.png', bbox_inches='tight')
