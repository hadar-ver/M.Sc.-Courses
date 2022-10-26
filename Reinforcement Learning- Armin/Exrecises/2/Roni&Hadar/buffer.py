import random
from collections import namedtuple

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'not_done'))


class ReplayBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        # TODO

        #Option #1: 
        len_memory = len(self.memory)
        if len_memory < self.capacity:  # if the capacity is greater then the len of memory list - memory list will be extended until is reached to its capacity limit
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)  
        self.position = (self.position + 1) % self.capacity

        #Option #2: 
        # self.memory.append(None)
        # self.memory[self.position] = Transition(*args)
        # if len(self.memory) > self.capacity:
        #     del self.memory[0]

    def sample(self, batch_size):
        # TODO
        # create random batch - random.sample(population, k, *, counts=None)¶
        return random.sample(self.memory, batch_size)


    def __len__(self):
        return len(self.memory)



