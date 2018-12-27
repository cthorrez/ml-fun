import torch
import numpy as np

class Buffer():
    def __init__(self, max_size, s_dim, a_dim):
        self.max_size = max_size
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.s_buffer = np.zeros((max_size, s_dim), dtype=np.float32)
        self.a_buffer = np.zeros((max_size, a_dim), dtype=np.float32)
        self.s_p_buffer = np.zeros((max_size, s_dim), dtype=np.float32)
        self.r_buffer = np.zeros((max_size), dtype=np.float32)
        self.done_buffer = np.zeros((max_size), dtype=np.float32)
        self.size = 0
        self.next = 0

    def add_tuple(self, s, a, r, s_p, done):
        insert_idx = self.next % self.max_size
        self.s_buffer[insert_idx,:] = s
        self.a_buffer[insert_idx,:] = a
        self.s_p_buffer[insert_idx,:] = s_p
        self.r_buffer[insert_idx] = r
        self.done_buffer[insert_idx] = done
        self.next += 1
        self.size = max(self.size, insert_idx)

    def sample(self, batch_size):
        batch_size = min(batch_size, self.size)
        idxs = torch.randint(low=0, high=self.size, size=(batch_size,))
        s = self.s_buffer[idxs,:]
        a = self.a_buffer[idxs,:]
        s_p = self.s_p_buffer[idxs,:]
        r = self.r_buffer[idxs]
        done = self.done_buffer[idxs]

        return torch.tensor(s, requires_grad=False), \
               torch.tensor(a, requires_grad=False), \
               torch.tensor(r, requires_grad=False), \
               torch.tensor(s_p, requires_grad=False), \
               torch.tensor(done, requires_grad=False)
        

