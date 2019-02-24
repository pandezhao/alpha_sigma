import numpy as np
import random
import sys
import utils

num2char = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h", 8: "i", 9: "j", 10: "k", 11: "l", 12: "m",
            13: "n", 14: "o", 15: "p", 16: "q", 17: "r", 18: "s", 19: "t", 20: "u"}
char2num = {"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7,"i":8,"j":9,"k":10,"l":11,"m":12,"n":13,"o":14}

# def tostring(num):
#     if num in num2char.keys():
#         return num2char[num]
#     elif num in char2num.keys():
#         return char2num[num]
#     else:
#         raise ValueError("board location should be within range [0, 14], if you modify the board size, you should also modify function 'tostring' in MCTS.py")

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
            self.child[node_name] = [node, node_operation]
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
    def __init__(self, board_size=13, simulation=1800):
        # when we are talking about win or lose, we are talking about first player(black stone player's win or lose)
        self.board_size = board_size
        self.simulation_number = simulation

        self.node_record = []
        self.node_record.append({"": node("")})

        self.current_node = self.node_record[0][""]
        self.current_step = 0
        self.current_expand = False
        self.current_expand_player = None
        self.current_player = 1 ## 1 represent black player, 0 represent white player.


    def restart(self):
        self.current_node = self.node_record[0][""]
        self.current_step = 0
        self.current_expand = False
        self.current_expand_player = None
        self.current_player = 1

    def generate_new_state(self, old_name, step):  # given the old name and step, give back the new name
        if self.current_player:
            step = "B" + num2char[step[0]]+num2char[step[1]]
        else:
            step = "W" + num2char[step[0]]+num2char[step[1]]
        for i in range(0, len(old_name), 3):
            if old_name[i+1]>step[1] or (old_name[i+1]==step[1] and old_name[i+2]>step[2]):
                new_name = old_name[:i] + step + old_name[i:]
                return new_name
        new_name = old_name + step
        return new_name

    def game_step(self, last_step, game_continue):
        if last_step is None:
            return self.current_node.child
        if game_continue == None:
            self.current_node.back_prop(0)
        if not game_continue:  # 如果胜利，那么就该执行的是反向传播操作，else里边是标准的树查找操作。
            if self.current_expand:
                if self.current_expand_player == self.current_player:
                    self.current_node.back_prop(1)
                else:
                    self.current_node.back_prop(-1)
            else:
                self.current_node.back_prop(1)
        else:  # 如果current expand是True的话，根本不会进入game step这个函数里边，所以根本不需要考虑current expand如何如何。
            # 这里我们先决定当前node的名字。
            new_name = self.generate_new_state(old_name=self.current_node.state, step=last_step)
            if new_name in self.current_node.child.keys():  # 不在这里有两种情况，一种是这个node是全新的，以前没有的，另一种是这个node虽然已经在总的库里，但是不在这个库里。
                tmp = self.current_node
                self.current_node = self.current_node.child[new_name]  # 因为我们player翻转的部分在外部坐过了，所以没必要在这里在做了。
                self.current_node.parent = tmp  # 对于一个node来说，parent node应该是动态的
            else:
                if self.current_step >= len(self.node_record):  # 长都不够长，肯定是压根就没有这个node了。
                    new_node = node(new_name)
                    new_node.parent = self.current_node
                    self.node_record.append({new_name:new_node})
                    self.current_node.add_child(node=self.node_record[self.current_step][new_name], node_name=new_name, node_operation=last_step)  # 这行我故意这么写，因为current_step不可能比node——record大过2。
                    self.current_expand = True    # 拓展过了，
                    self.current_expand_player = self.current_player
                    self.current_node = self.current_node.child[new_name]
                    """
                        初步认为，self.current_expand_player = self.current_player。这里是没有问题的。
                    """

                elif new_name in self.node_record[self.current_step].keys():  # 新的node已经有了，但是没有加到current_node的child里边，这里我认为是不算拓展过的了。
                    self.current_node.add_child(node=self.node_record[self.current_step][new_name], node_name=new_name, node_operation=last_step)
                    self.current_node.child[new_name].parent = self.current_node
                    self.current_node = self.current_node.child[new_name]
                else:  # 压根没有新的node，但是长倒是够长了。我们在这里新建。
                    new_node = node(new_name)
                    self.node_record[self.current_step][new_name] = new_node
                    self.current_node.add_child(node=self.node_record[self.current_step][new_name], node_name=new_name, node_operation=last_step)
                    self.current_expand = True
                    self.current_expand_player = self.current_player
                    self.current_node.child[new_name].parent = self.current_node
                    self.current_node = self.current_node.child[new_name]
        return self.current_node.child

    def Simulation_step(self, last_step, board, game_continue):  # used to communicate with the real game environment.
        # 这里主要的目的是输入上一个step进行之后的结果
        '''
        大致的流程图是这样。
        1. environment输入None action
        2. 因为输入是None action,所以在MCTS这里随机选取一步，不进入game step。直接将action返回给环境
        3. 环境根据返回的action，给出输赢结果，并将last step，输赢结果和valid move list输入MCTS step
        4. MCTS step获得输赢结果和last step，输入进game step函数中，根据child node和valid move list选择出合适的下一步move，返回给环境
        。。。
        X: 环境返回的last move和输赢，输赢已定，所以将last step和输赢输入进game step函数中，MCTS search一次结束。
        '''
        if type(last_step)==np.ndarray:
            self.current_step += 1
        if type(board) != np.ndarray:
            board = np.zeros([self.board_size, self.board_size])
        if game_continue:  #如果上一步分出了胜负，我们就不应该再在这里提前翻转了。
            self.current_player = 1 - self.current_player
        if not self.current_expand:
            if game_continue:
                child = self.game_step(last_step, game_continue=game_continue)
                max_v = -sys.maxsize
                max_node = None

                child_node_action = []
                for child_node in child.values():
                    if not board[child_node.action[0], child_node.action[1]]:
                        tmp = child_node.value + np.sqrt(np.log2(self.current_node.counter)/child_node.counter)
                        child_node_action.append(child_node.action)
                        if tmp > max_v:
                            max_node = child_node
                            max_v = tmp
                if max_v > np.sqrt(np.log2(self.current_node.counter)):
                    return max_node.action
                else:
                    last_step = self.random_step(board=board, child_node_action=child_node_action)# 这里加一个随机走路的函数
                    return last_step

            else:
                self.game_step(last_step, game_continue=game_continue)
        elif not game_continue:
            self.game_step(last_step, game_continue=game_continue)
        else:
            return self.random_step(board=board, child_node_action=[])
        if not game_continue:
            self.restart()
            return None

    def MCTS_main(self, games):

        for _ in range(self.simulation_number):
            self.Simulation_step()
            

    def random_step(self, board, child_node_action):
        random_list = list(np.argwhere(board == 0))
        if len(random_list) == len(child_node_action):
            return random.choice(random_list)
        if len(child_node_action)!=0:
            tmp = utils.step_child_remove(random_list, child_node_action)
        else:
            tmp = list(np.argwhere(board == 0))
        result = random.choice(tmp)
        if board[result[0], result[1]]!=0:
            print("here i stop")
        return result


    # def game_step(self, step, game_continue=True):  # used to expand MCTS and other stuff
    #     if step:  # for first step, here should be a None.
    #         self.current_player = 1 - self.current_player
    #         tmp=self.current_node.state
    #         new_state = tmp + tostring(step[0])+tostring(step[1])
    #         self.current_step += 1
    #         if len(self.node_record) > self.current_step:  ## 如果我现在还没有探索到了最深的步的话
    #             if new_state in self.node_record[self.current_step].keys():
    #                 self.current_node = self.node_record[self.current_step][new_state]
    #             else:
    #                 self.current_node.add_child(step)
    #                 self.node_record[self.current_step][new_state] = self.current_node.child[new_state]
    #                 self.current_expand = True
    #                 self.current_node = self.current_node.child[new_state]
    #         else:
    #             self.current_node.add_child(step)
    #             self.node_record.append({})
    #             self.node_record[self.current_step][new_state] = self.current_node.child[new_state]
    #             self.current_expand = True
    #             self.current_node = self.current_node.child[new_state]
    #         if self.current_expand: # 如果我到达这里，说明之前并没有expand，但是现在扩展了。所以current node是最后一个扩展的node，我以后的输赢从这里拓展
    #             # 如果我以后的胜利者和现在是同一个玩家，则back prop一个（1），否则back prop一个（-1）
    #             self.current_expand_player = self.current_player
    #
    #         if not game_continue:  # if this game has a result, we would back prop from the last expand node.
    #             if self.current_expand:
    #                 if self.current_expand_player == self.current_player:
    #                     self.current_node.back_prop(1)
    #                 else:
    #                     self.current_node.back_prop(-1)
    #             else:
    #                 self.current_node.back_prop(1)
    #             return None
    #         else:
    #             return self.current_node.child
    #     else:
    #         return self.current_node.child