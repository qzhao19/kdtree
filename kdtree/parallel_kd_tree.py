import ctypes 
import os
import multiprocessing as processing
from .kd_tree import KDTree

def __num_processors():
    if os.name == 'nt': 
        # Windows
        return int(os.getenv('NUMBER_OF_PROCESSORS'))
    else: 
        # glibc (Linux, *BSD, Apple)
        get_nprocs = ctypes.cdll.libc.get_nprocs
        get_nprocs.restype = ctypes.c_int
        get_nprocs.argtypes = []
        return get_nprocs()


def __remote_query(rank, qin, qout, kdtree, data, k, radius):
    while 1:
        # read input queue (block until data arrives)
        nc, data = qin.get()
        # process data
        knn = kdtree.query(data, k)
        # write to output queue
        qout.put((nc, knn))

def __remote_query_radius(rank, qin, qout, kdtree, data, k, radius):
    while 1:
        # read input queue (block until data arrives)
        nc, data = qin.get()
        # process data
        rnn = kdtree.query_radius(data, radius)
        # write to output queue
        qout.put((nc, rnn))


class ParallelKDTree(object):
    def __init__(self, data, leaf_size = 10, splitter = None, chunk_size = None):
        self.data = data
        if self.data.size == 0:
            raise ValueError("X is an empty array")

        self.num_samples = self.data.shape[0]
        self.num_features = self.data.shape[1]

        if leaf_size < 1:
            raise ValueError("leaf_size must be greater than or equal to 1")
        self.leaf_size = leaf_size
        self.splitter = splitter
        self.chunk_size = chunk_size

        # get the number of processors based on different platforms
        self.num_proc = __num_processors()

        # compute chunk size
        if not chunk_size:
            self.chunk_size = self.num_samples / (4 * self.num_proc)
        else:
            self.chunk_size = chunk_size
        self.chunk_size = 100 if self.chunk_size < 100 else self.chunk_size

        self.kdtree = KDTree(data, self.leaf_size, self.splitter)
    
    def _parallelize_job(self, job, data, k, radius):
        # set up a pool of processes
        qin = processing.Queue(maxsize = self.num_samples/self.chunk_size)
        qout = processing.Queue(maxsize = self.num_samples/self.chunk_size)
        processor = []
        for rank in range(self.num_proc):
            processing.Process(target=job, args=(rank, qin, qout, self.kdtree, data, k, radius))
        for p in processor: 
            p.start()
        
        # put data chunks in input queue
        cur, nc = 0, 0
        while 1:
            _data = data[:,cur:cur + self.chunk_size]
            if _data.shape[1] == 0: break
            qin.put((nc,_data))
            cur += self.chunk_size
            nc += 1
        
         # read output queue
        nn = []
        while len(nn) < nc:
            nn += [qout.get()]
        
         # avoid race condition
        _nn = [n for i,n in sorted(nn)]
        knn = []
        for tmp in _nn:
            knn += tmp
        # terminate workers
        for p in processor: 
            p.terminate()
        return knn

    def query(self, data, k):
        self._parallelize_job(__remote_query, data, k = k, radius = None)
        

    def query_radius(self, data, radius):
        self._parallelize_job(__remote_query_radius, data, k = None, radius = radius)

