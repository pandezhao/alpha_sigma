import numpy as np
import pickle

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
