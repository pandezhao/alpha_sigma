import torch
import numpy as np

class main_process:
    def __init__(self, board_size=15, AI = None):
        self.board_size = board_size
        self.board = np.zeros([self.board_size+8, self.board_size+8])
        self.current_player = 1
        self.passed = 0
        if AI:
            self.AI = "random"

    def renew(self):
        self.board = np.zeros([self.board_size+8, self.board_size+8])
        self.current_player = 1
        self.passed = 0

    def vs_AI(self):
        pass

    def which_player(self):
        return self.current_player

    def current_board_state(self, raw=False):
        if raw:
            return self.board
        else:
            return self.board[4:self.board_size+4, 4:self.board_size+4]

    def simulate_reset(self, board_state):  # 这里还可以添加一个检测输入棋盘是否为一局胜负以分的board的算法，但是由于不影响强化学习，所以暂且搁置
        if type(board_state) != np.ndarray: # 而且还有一部分原因也是因为没有找到一个合适的算法。
            raise ValueError("board_state must be a np array")
        if board_state.shape[0]!=self.board_size+8 or board_state.shape[1]!=self.board_size+8:
            raise ValueError("board size is different from the given size")
        self.board = np.array(board_state, copy=True)
        step_count, black_count, white_count = 0, 0, 0
        for hang in board_state:
            for lie in hang:
                if lie == 1:
                    step_count += 1
                    black_count += 1
                elif lie == -1:
                    step_count += 1
                    white_count += 1
                elif lie != 0:
                    raise ValueError("We got some value wrong in the input board_state, the value in board must be 0, 1, -1")
        if black_count == white_count:
            self.current_player = 1
            self.passed = step_count
        elif black_count == white_count + 1:
            self.current_player = -1
            self.passed = step_count
        else:
            raise ValueError("The input board state is not a standard five stone game board, the number of black stone"
                             " and white stone is incorrect")

    def step(self, place):  # 这里还需要增加下board state的厚度，因为原版的输入到神经网络里免得board state可并不是只有一个2D matrix。
        self.passed += 1    # 所以我们需要增加下厚度。
        if not self.board[place[0]+4, place[1]+4]:
            self.board[place[0]+4, place[1]+4] = self.current_player
            self.current_player = -self.current_player
            self.last_step = [place[0]+4, place[1]+4]
            if self.check_win():
                return False, self.board[4:self.board_size+4, 4:self.board_size+4]
            elif self.passed == self.board_size * self.board_size:
                return None, self.board[4:self.board_size+4, 4:self.board_size+4]
            return True, self.board[4:self.board_size+4, 4:self.board_size+4]
        else:
            raise ValueError("here already has a stone, you can't please stone on it")

    def check_win(self):
        if self.board[self.last_step[0], self.last_step[1]] == 1:
            x_tmp, y_tmp = self.last_step[0], self.last_step[1]
            for i in range(5):
                if sum(self.board[x_tmp-4+i:x_tmp+1+i, y_tmp])==5:
                    return True
                elif sum(self.board[x_tmp, y_tmp-4+i:y_tmp+1+i])==5:
                    return True
                elif self.board[x_tmp+i-4,y_tmp+i-4]+self.board[x_tmp+i-3,y_tmp+i-3]+self.board[x_tmp+i-2,y_tmp+i-2]+\
                        self.board[x_tmp+i-1,y_tmp+i-1]+self.board[x_tmp+i,y_tmp+i]==5:
                    return True
                elif self.board[x_tmp+i-4,y_tmp-i+4]+self.board[x_tmp+i-3,y_tmp-i+3]+self.board[x_tmp+i-2,y_tmp-i+2]+\
                        self.board[x_tmp+i-1,y_tmp-i+1]+self.board[x_tmp+i,y_tmp-i]==5:
                    return True
        elif self.board[self.last_step[0], self.last_step[1]] == -1:
            x_tmp, y_tmp = self.last_step[0], self.last_step[1]
            for i in range(5):
                if sum(self.board[x_tmp - 4 + i:x_tmp + 1 + i, y_tmp]) == -5:
                    return True
                elif sum(self.board[x_tmp, y_tmp - 4 + i:y_tmp + 1 + i]) == -5:
                    return True
                elif self.board[x_tmp + i - 4, y_tmp + i - 4] + self.board[x_tmp + i - 3, y_tmp + i - 3] + self.board[
                    x_tmp + i - 2, y_tmp + i - 2] + \
                        self.board[x_tmp + i - 1, y_tmp + i - 1] + self.board[x_tmp + i, y_tmp + i] == -5:
                    return True
                elif self.board[x_tmp + i - 4, y_tmp - i + 4] + self.board[x_tmp + i - 3, y_tmp - i + 3] + self.board[
                    x_tmp + i - 2, y_tmp - i + 2] + \
                        self.board[x_tmp + i - 1, y_tmp - i + 1] + self.board[x_tmp + i, y_tmp - i] == -5:
                    return True
        return False