import torch
import numpy as np

class main_process:
    def __init__(self, board_size=15, AI = None):
        self.board_size = board_size
        self.board = np.zeros([self.board_size+8, self.board_size+8])
        self.valid_backup = []
        self.current_player = 1
        self.passed = 0
        if AI:
            self.AI = "random"

        self.games_counter = 0
        self.games_black_win_counter = 0
        self.games_white_win_counter = 0
        self.no_win_games_counter = 0

        self.tmp_black_win = 0
        self.tmp_white_win = 0
        self.tmp_heqi = 0

        self.board_record = []
        self.place_record = []

    def renew(self):
        self.board = np.zeros([self.board_size+8, self.board_size+8])
        self.current_player = 1
        self.passed = 0

    def clear_tmp(self):
        self.tmp_black_win = 0
        self.tmp_white_win = 0
        self.tmp_heqi = 0

    def vs_AI(self):
        pass

    def which_player(self):
        return self.current_player

    def step(self, place):
        self.passed += 1
        self.place_record.append(place)
        if not self.board[place[0]+4, place[1]+4]:
            self.board[place[0]+4, place[1]+4] = self.current_player
            self.current_player = -self.current_player
            self.last_step = [place[0]+4, place[1]+4]
            if self.check_win():
                self.games_counter += 1
                if self.current_player == 1:
                    self.games_black_win_counter += 1
                    self.tmp_black_win += 1
                else:
                    self.games_white_win_counter += 1
                    self.tmp_white_win += 1

                return False, self.board[4:self.board_size+4, 4:self.board_size+4]
            elif self.passed == self.board_size * self.board_size:
                self.games_counter += 1
                self.no_win_games_counter += 1
                self.tmp_heqi += 1

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