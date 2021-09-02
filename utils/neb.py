
# NEB code: https://gitlab.com/ase/ase/-/blob/f153e5aeef80bda03284de9740967c47ba81232b/ase/neb.py

from ase import Atoms
from ase.neb import NEB
from ase.constraints import FixAtoms
from ase.optimize import BFGS

from utils.alloymap import AlloyGenerator

import numpy as np
import matplotlib.pyplot as plt

import os, sys

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

def get_neb_energy(initial_state, goal_state, active_dist = 7):
    """
        Outputs the NEB barrier between two inputtet states
    """
    # Read initial and final states:
    initial = initial_state
    final = goal_state

    # Specify agent atom
    n_atoms = len(initial)
    agent_atom = n_atoms - 1

    constrain_bottom = True

    # Number of images
    n_im = 5

    # NEB spring konstant
    k_spring = 0.1

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
        image.calc = EMT()
        if constrain_bottom:
            image.set_constraint(image.constraints + [bottom_constraint])

    # Interpolate linearly the potisions of the three middle images:
    neb.interpolate()

    # Optimize and save NEB optimization as traj
    # optimizer = MDMin(neb, trajectory='A2B.traj')
    optimizer = BFGS(neb)
    optimizer.run(fmax=0.05)

    # Energy barrier
    start_energy = neb.images[0].get_potential_energy()
    neb_energies = [neb.images[i].get_potential_energy() - start_energy for i in range(n_im)]
    max_id = np.array(neb_energies).argmax()

    neighbor_ids = np.argsort(
        neb.images[max_id].get_distances(
            a=agent_atom,
            indices=list(range(n_atoms)),
            mic=True
        )
    )[1:3]

    bridge_atoms = neb.images[max_id].get_atomic_numbers()[neighbor_ids]
    is_Ag = bridge_atoms == 47
    n_Ag = is_Ag.sum()

    neighbor_mean_height = neb.images[max_id].get_positions()[neighbor_ids, 2].mean()
    height = neb.images[max_id].get_positions()[agent_atom, 2] - neighbor_mean_height

    return max(neb_energies), bridge_atoms, n_Ag, height



