import random
from collections import namedtuple

class ReplayMemoryMonteCarlo(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.Transition = namedtuple('Transition',
            ('state', 'action', 'ret', 'agent_atom', 'B', 'n_surf'))
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
