
# https://wiki.fysik.dtu.dk/ase/tutorials/neb/diffusion.html

import numpy as np
from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton, BFGS
from ase.visualize import view

from ase import Atoms
from ase.io import Trajectory
from ase.io import write

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

from ase.utils.structure_comparator import SymmetryEquivalenceCheck
from itertools import chain

import random
from collections import namedtuple


class AlloyGenerator(object):

    def __init__(self, n_alloys = 1000, n_free: int=18, atomic_numbers: list=[13, 47],
        adsorbate: str='Au', alloy_type: str='Ag', fmax: float=0.05):

        self.n_free = n_free
        self.atomic_numbers = atomic_numbers
        self.adsorbate = adsorbate
        self.alloy_type = alloy_type
        self.fmax = fmax
        self.slab_dim = (3,3,3)
        self.memory = []
        self.position = 0
        #self.n_alloys = n_alloys # don't use anymore if we just sample using no alloy_batch
        #self.alloy_batch = list(product(atomic_numbers, repeat=n_free))
        #np.random.shuffle(self.alloy_batch)

    def sample(self, batch_size=1, sym_check=False):
        """
            Sample new alloy and check whether it is equivalent to earlier alloys
        """

        alloy_atoms = np.random.choice(a=self.atomic_numbers, size=self.n_free)
        self.memory.append(alloy_atoms)

        #alloy_atoms = random.sample(self.alloy_batch, batch_size)
        #if sym_check:
        #    for done_alloy in self.memory:
        #        symmetric = self.check_symmetry(alloy_atoms, done_alloy)
        #        if symmetric:
        #            
        
        return alloy_atoms


    def check_symmetry(self, alloy_atoms1, alloy_atoms2):
        """
            Creates specified alloys and checks if they are symmetry equivalent
        """
        # Slab a
        a = fcc100('Al', size=(3,3,2), periodic=True)
        a.set_atomic_numbers(alloy_atoms1)

        # Slab b
        b = a.copy()
        b.set_atomic_numbers(alloy_atoms2)

        # Compare
        comp = SymmetryEquivalenceCheck()

        return comp.compare(a,b)


    def get_relaxed_alloy_map(self, alloy_atoms, map):

        # alloy_atoms = self.sample(batch_size=1)

        # Build slab and specify atoms
        slab = fcc100(self.alloy_type, size=self.slab_dim)
        atomic_numbers = slab.get_atomic_numbers()
        atomic_numbers[[atom.tag < self.slab_dim[2] for atom in slab]] = alloy_atoms
        slab.set_atomic_numbers(atomic_numbers)

        # Add adsorbate
        add_adsorbate(slab, self.adsorbate, 1.7, 'ontop')
        slab.center(axis=2, vacuum=4.0)

        # Fix bottom layer:
        mask = [atom.tag > 2 for atom in slab]
        slab.set_constraint(FixAtoms(mask=mask))

        # Use EMT potential:
        slab.calc = EMT()

        # Make copy of slab before adjusting agent atom
        slab_b = slab.copy()

        # Half an interatomic distance is given by dx and dy
        dx = slab.get_cell()[0, 0] / (2 * self.slab_dim[0])
        dy = slab.get_cell()[1, 1] / (2 * self.slab_dim[1])

        # Specify initial location of agent atom 
        if map == 0:
            slab[-1].x += 3 * dx
            slab[-1].y += 2 * dy
        elif map == 1:
            slab[-1].x += 2 * dx
            slab[-1].y += 2 * dy
        elif map == 2:
            slab[-1].x += 3 * dx
            slab[-1].y += 1 * dy
        elif map == 3:
            slab[-1].x += 2 * dx
            slab[-1].y += 1 * dy
        elif map == 4:
            slab[-1].x += 1 * dx
            slab[-1].y += 1 * dy
        elif map == 5:
            slab[-1].x += 1 * dx
            slab[-1].y += 2 * dy


        # NOTE: Do (4,4,3) to have more maps

        # Initial state:
        qn = QuasiNewton(slab)
        qn.run(fmax=self.fmax)

        # Final state (always same location):
        slab_b.calc = EMT()
        slab_b[-1].x += 3 * dx
        slab_b[-1].y += 3 * dy

        qn = QuasiNewton(slab_b)
        qn.run(fmax=self.fmax)

        return slab, slab_b


    def get_relaxed_single(self, alloy_atoms, hollow_site):

        # alloy_atoms = self.sample(batch_size=1)

        # Build slab and specify atoms
        slab = fcc100(self.alloy_type, size=self.slab_dim)
        atomic_numbers = slab.get_atomic_numbers()
        atomic_numbers[[atom.tag < self.slab_dim[2] for atom in slab]] = alloy_atoms
        slab.set_atomic_numbers(atomic_numbers)

        # Add adsorbate
        add_adsorbate(slab, self.adsorbate, 1.7, 'ontop')
        slab.center(axis=2, vacuum=4.0)

        # Fix bottom layer:
        mask = [atom.tag > 2 for atom in slab]
        slab.set_constraint(FixAtoms(mask=mask))

        # Use EMT potential:
        slab.calc = EMT()

        # Half an interatomic distance is given by dx and dy
        dx = slab.get_cell()[0, 0] / (2 * self.slab_dim[0])
        dy = slab.get_cell()[1, 1] / (2 * self.slab_dim[1])

        # Specify initial location of agent atom 

        if hollow_site == 'right':
            slab[-1].x += 3 * dx
            slab[-1].y += 1 * dy
        elif hollow_site == 'up':
            slab[-1].x += 1 * dx
            slab[-1].y += 3 * dy


        # Initial state:
        qn = QuasiNewton(slab)
        qn.run(fmax=self.fmax)

        return slab

