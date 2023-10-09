import ctypes 
import os
try:
    import multiprocessing as processing
except:
    import processing

from .kd_tree import KdTree

def __num_processors():
    if os.name == 'nt': # Windows
        return int(os.getenv('NUMBER_OF_PROCESSORS'))
    else: # glibc (Linux, *BSD, Apple)
        get_nprocs = ctypes.cdll.libc.get_nprocs
        get_nprocs.restype = ctypes.c_int
        get_nprocs.argtypes = []
        return get_nprocs()


def __remote_process(qin, qout, kdtree, data, k):
    while 1:
        # read input queue (block until data arrives)
        nc, data = qin.get()
        # process data
        knn = kdtree.query(data, k)
        # write to output queue
        qout.put((nc,knn))


class ParallelKdTree(object):
    def __init__(self) -> None:
        pass
    
    
    