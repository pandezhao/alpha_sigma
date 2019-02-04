from network import neuralnetwork
from utils import random_stack

class main:
    def __init__(self):
        self.NN = neuralnetwork(11)
        self.database = random_stack(length=1000, clear_size=100)
