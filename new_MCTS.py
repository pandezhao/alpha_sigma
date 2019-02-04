import numpy as np
import random
import utils


num2char = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h", 8: "i", 9: "j", 10: "k", 11: "l", 12: "m",
            13: "n", 14: "o", 15: "p", 16: "q", 17: "r", 18: "s", 19: "t", 20: "u"}

class edge:
    def __init__(self, action, parent_node, priorP):

class node:
    def __init__(self, state, parent):
        self.state_name = state
        self.parent = parent
        self.value = 0
        self.counter = 0
        self.child = {}

    def add_child(self, state, node, action, priorP):
        self.child[state] = {"node":node, "action":action, "prior":priorP}

    def backup(self, v):
        self.value += v
        self.counter += 1
        if self.parent:
            self.parent.backup(-v)

class MCTS:
    def __init__(self, board_size=11, simulation_per_step=1800, neural_network=None, init_state="", init_node=None):

        self.board_size = board_size
        self.s_per_step = simulation_per_step
        self.database = [[node(init_state, init_node)]]
        self.current_node = self.database[0][0]
        self.NN = neural_network

    def simulation(self, state):
        expand = False
        this_node = self.current_node
        valid_move = utils.valid_move(state)
        while not expand:
            state_prob, state_v = self.NN.eval(state)

            for move in valid_move:
                this_node.add_child(node(utils.generate_new_state()), move, state_prob[move])

    def game_step(self):