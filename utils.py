import numpy as np
import pickle

import copy
import random

num2char = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h", 8: "i", 9: "j", 10: "k", 11: "l", 12: "m",
            13: "n", 14: "o", 15: "p", 16: "q", 17: "r", 18: "s", 19: "t", 20: "u"}

char2num = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12,
            "n": 13, "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19, "u": 20}

temperature = 0.1

class distribution_calculater:
    def __init__(self, size):
        self.map = {}
        self.order = []
        for i in range(size):
            for j in range(size):
                name = num2char[i]+num2char[j]
                self.order.append(name)
                self.map[name] = 0

    def push(self, key, value):
        self.map[key] = value

    def get(self):
        result = []
        for key in self.order:
            result.append(self.map[key])
            self.map[key] = 0
        return result

def step_child_remove(board_pool, child_pool):
    i = 0
    while i<len(board_pool) and len(child_pool) != 0:
        j = 0
        while j<len(child_pool):
            if np.array_equal(board_pool[i], child_pool[j]):
                board_pool.pop(i)
                child_pool.pop(j)
                i -= 1
                break
            else:
                j += 1
        i+=1
    return board_pool

def write_file(object, file_name):
    filewriter = open(file_name, 'wb')
    pickle.dump(object, filewriter)
    filewriter.close()

def read_file(file_name):
    filereader = open(file_name, 'rb')
    object = pickle.load(filereader)
    filereader.close()
    return object

def move_to_str(action):
    return num2char[action[0]] + num2char[action[1]]

def str_to_move(str):
    return np.array([char2num[str[0]], char2num[str[1]]])

def valid_move(state):
    return list(np.where(state!=0))

def generate_new_state(old_name, step, current_player):
    if current_player == 1:
        step = "B" + num2char[step[0]] + num2char[step[1]]
    else:
        step = "W" + num2char[step[0]] + num2char[step[1]]
    for i in range(0, len(old_name), 3):
        if old_name[i+1]>step[1] or (old_name[i+1]==step[1] and old_name[i+2]>step[2]):
            new_name = old_name[:i] + step + old_name[i:]
            return new_name
    new_name = old_name + step
    return new_name

class random_stack:
    def __init__(self, length=1000, clear_size=100):
        self.data = []
        self.length = length
        self.clear_size = clear_size

    def isEmpty(self):
        return len(self.data) == 0

    def push(self, item):
        self.data.append(item)
        if len(self.data)>= (self.length+self.clear_size):
            self.data = self.data[self.clear_size:]

    def seq(self):
        return copy.deepcopy(self.data)

    def random_seq(self):
        tmp = copy.deepcopy(self.data)
        random.shuffle(tmp)
        return tmp