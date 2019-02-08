import numpy as np
import random
import utils
from five_stone_game import main_process

num2char = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h", 8: "i", 9: "j", 10: "k", 11: "l", 12: "m",
            13: "n", 14: "o", 15: "p", 16: "q", 17: "r", 18: "s", 19: "t", 20: "u"}

class edge:
    def __init__(self, action, parent_node, priorP, MCTS_pointer):
        self.action = action
        self.counter = 0
        self.parent_node = parent_node
        self.priorP = priorP
        self.MCTS_ptr = MCTS_pointer
        self.child_node = self.search_and_get_child_node()

    def backup(self, v):
        self.parent_node.backup(-v)

    def get_child(self):
        self.counter += 1
        return self.child_node

    def search_and_get_child_node(self):
        new_state_name = utils.generate_new_state(self.parent_node.get_state, self.action, self.parent_node.node_player)
        search_result = self.MCTS_ptr.search_node(new_state_name)
        if search_result:
            return search_result
        else:
            return node(new_state_name, self, 1 - self.parent_node.node_player)

class node:
    def __init__(self, state, parent, player):
        self.state_name = state
        self.parent = parent
        self.value = 0
        self.counter = 0
        self.child = {}

        self.node_player = player

    def get_state(self):
        return self.state_name

    def add_child(self, state, action, priorP):
        self.child[action] = edge(action=action, parent_node=self, priorP=priorP)

    def get_child(self, action):
        return self.child[action].get_child()

    def backup(self, v):
        self.value += v
        self.counter += 1
        if self.parent:
            self.parent.backup(v)

    def get_distribution(self): ## used to get the step distribution of current

        for key in self.child.keys():



class MCTS:
    def __init__(self, board_size=11, simulation_per_step=1800, neural_network=None, init_state="", init_node=None):

        self.board_size = board_size
        self.s_per_step = simulation_per_step
        self.database = [[node(init_state, init_node)]]
        self.current_node = self.database[0][0]
        self.NN = neural_network
        self.game_process = main_process(board_size=board_size)

    def search_node(self, node_name):


    def simulation(self, state):
        expand = False
        this_node = self.current_node
        valid_move = utils.valid_move(state)
        while not expand:
            state_prob, state_v = self.NN.eval(state)

            for move in valid_move:
                this_node.add_child(node(utils.generate_new_state()), move, state_prob[move])

    def game(self):

        game_continue = True
        while game_continue:
            self.simulation()
            action, distribution = self.current_node.get_distribution()
            action = max(distribution)  # some method to choose the largest action
            '''
            no finished here
            '''
            game_continue, state = self.game_process.step(action)

            self.current_node =
