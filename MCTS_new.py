import numpy as np
import random
import sys
import utils
from five_stone_game import main_process

num2char = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h", 8: "i", 9: "j", 10: "k", 11: "l", 12: "m",
            13: "n", 14: "o", 15: "p", 16: "q", 17: "r", 18: "s", 19: "t", 20: "u"}
char2num = {"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7,"i":8,"j":9,"k":10,"l":11,"m":12,"n":13,"o":14}

class node:
    def __init__(self, state):
        self.state = state
        self.parent = None
        self.child = {}
        self.counter = 1
        self.value = 0
        self.child_value = 0

    def add_child(self, node, node_name, node_operation):
        if node_name not in self.child.keys():
            self.child[node_name] = {"node": node, "action": node_operation}
        else:
            raise KeyError("We are adding a child that has exists, how could this happened? There must be bugs.")

    def back_prop(self, v):  # 如果当前胜利者编号是1,也就是说是黑棋胜利，那么back prop的应该是 正v，如果当前胜利者是白旗，反向传播的是 负v。
        self.counter += 1
        self.child_value += v
        if self.child:
            self.value = self.child_value / len(self.child)
        else:
            self.value = self.child_value
        if self.parent:
            self.parent.back_prop(-v)

class MCTS:
    def __init__(self, board_size=11, simulation_times=1800, UCB_p=1, games=None):
        self.board_size = board_size
        self.simulation_times = simulation_times
        self.UCB_p = UCB_p
        self.game_process = main_process(board_size=board_size)
        self.games = None

        self.node_record = []
        self.node_record.append({"": node("")})

        self.current_root = self.node_record[0][""]

        self.simulation_node = self.current_root
        self.simulation_step = 0
        self.simulation_expand = False
        self.simulation_expand_player = None
        self.simulation_player = 1

    def restart(self):
        self.current_root = self.node_record[0][""]
        self.simulation_node = self.current_root
        self.simulation_step = 0
        self.simulation_expand = False
        self.simulation_expand_player = None
        self.simulation_player = 1

    def re_simulation(self, selected_move):
        self.simulation_node = self.current_root
        self.simulation_step = len(selected_move)
        self.simulation_expand = False
        self.simulation_expand_player = None
        self.simulation_player = 1 - len(selected_move)%2

    def simulation(self, selected_move):


        self.re_simulation(selected_move)

    def MCTS_main(self):
        selected_move = []
        while True:
            game_continue = True
            for move in selected_move:
                game_continue, board_state = self.game_process.step(move)
            if not game_continue:  # 当我们搜索到最优解是从头下棋到尾的时候，我们就停止搜索。
                return
            for _ in range(self.simulation_times):
                self.simulation(board_state)
            child = self.select_child_value(self.current_root)
            selected_move.append(child["action"])
            self.current_root = child["node"]

    def select_child_UCB(self, board, node):  # 用于根据UCB公式探索，主要应用在函数simulation种
        max_v = -sys.maxsize
        max_node_action = None

        child_node_action = []
        for child in node.child.values():
            tmp = child["node"].value + np.sqrt(np.log2(node.counter)/child["node"].counter)
            child_node_action.append(child["action"])
            if tmp > max_v:
                max_node_action = child["action"]
                max_v = tmp
        if max_v > np.sqrt(np.log2(node.counter)):
            return max_node_action
        else:
            random_node_step = self.random_step(board=board, child_node_action=child_node_action)
            return random_node_step


    def select_child_value(self, node):  # 用于确定下一个root node
        max_v = -sys.maxsize
        max_node = None

        for child in node.child.values():
            if child["node"].value > max_v
                max_v = child["node"].value
                max_node = child
        return max_node

    def random_step(self, board, child_node_action):
        random_list = list(np.argwhere(board == 0))
        if len(random_list) == len(child_node_action):
            return random.choice(random_list)
        if len(child_node_action) != 0:
            tmp = utils.step_child_remove(random_list, child_node_action)
        else:
            tmp = list(np.argwhere(board == 0))
        result = random.choice(tmp)
        if board[result[0], result[1]] != 0:
            print("here i stop")
        return result

    def generate_new_state(self, old_name, step, current_player):  # given the old name and step, give back the new name
        if current_player==1:
            step = "B" + num2char[step[0]]+num2char[step[1]]
        else:
            step = "W" + num2char[step[0]]+num2char[step[1]]
        for i in range(0, len(old_name), 3):
            if old_name[i+1]>step[1] or (old_name[i+1]==step[1] and old_name[i+2]>step[2]):
                new_name = old_name[:i] + step + old_name[i:]
                return new_name
        new_name = old_name + step
        return new_name