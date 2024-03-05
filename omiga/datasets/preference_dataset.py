import numpy as np
import torch
import h5py
from offline_dataset import ReplayBuffer


class PrefDataset(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        n_agents,
        env_name,
        data_dir,
        pref_dir,
        segment_length,
        max_feedbacks=int(1e5),
        device="cuda",
    ):
        self.max_feedbacks = max_feedbacks
        self.total_feedbacks = 0
        self.n_agents = n_agents
        self.env_name = env_name
        self.data_dir = data_dir
        self.pref_dir = pref_dir
        self.segment_length = segment_length
        self.device = device

        self.o1 = np.zeros((max_feedbacks, segment_length, n_agents, state_dim))
        self.s1 = np.zeros((max_feedbacks, segment_length, n_agents, state_dim))
        self.a1 = np.zeros((max_feedbacks, segment_length, n_agents, action_dim))
        self.r1 = np.zeros((max_feedbacks, segment_length, 1))
        self.mask1 = np.zeros((max_feedbacks, segment_length, 1))

        self.o2 = np.zeros((max_feedbacks, segment_length, n_agents, state_dim))
        self.s2 = np.zeros((max_feedbacks, segment_length, n_agents, state_dim))
        self.a2 = np.zeros((max_feedbacks, segment_length, n_agents, action_dim))
        self.r2 = np.zeros((max_feedbacks, segment_length, 1))
        self.mask2 = np.zeros((max_feedbacks, segment_length, 1))

        self.labels = np.zeros((max_feedbacks, 1))
        self.device = torch.device(device)
    
    def load(self):
        print('==========Data loading==========')
        data_file = self.data_dir + self.env_name + '.hdf5'
        # data_file = self.data_dir + 'test.hdf5'
        print('Loading from:', data_file)
        f = h5py.File(data_file, 'r')
        s = np.array(f['s'])
        o = np.array(f['o'])
        a = np.array(f['a'])
        r = np.array(f['r'])
        d = np.array(f['d'])
        f.close()

        print('==========PrefIndex loading==========')
        index = np.load(self.pref_dir)

        for i in range(index.shape[0]):
            ind1, ind2 = index[i][0], index[i][1]

            self.o1[i] = o[ind1 - self.segment_length : ind1]
            self.s1[i] = s[ind1 - self.segment_length : ind1]
            self.a1[i] = a[ind1 - self.segment_length : ind1]
            self.r1[i] = r[ind1 - self.segment_length : ind1]
            self.mask1[i] = 1 - d[ind1 - self.segment_length : ind1]

            self.o2[i] = o[ind2 - self.segment_length : ind2]
            self.s2[i] = s[ind2 - self.segment_length : ind2]
            self.a2[i] = a[ind2 - self.segment_length : ind2]
            self.r2[i] = r[ind2 - self.segment_length : ind2]
            self.mask2[i] = 1 - d[ind2 - self.segment_length : ind2]

            # script theacher
            self.labels[i] = 1.0 * (self.r1[i].sum(0) < self.r2[i].sum(0))
        
        self.total_feedbacks = index.shape[0]
    
    def sample(self, batch_size):
        ind = np.random.randint(0, self.total_feedbacks, size=batch_size)
        return (
            torch.FloatTensor(self.o1[ind]).to(self.device),
            torch.FloatTensor(self.s1[ind]).to(self.device),
            torch.FloatTensor(self.a1[ind]).to(self.device),
            torch.FloatTensor(self.mask1[ind]).to(self.device),
            torch.FloatTensor(self.o2[ind]).to(self.device),
            torch.FloatTensor(self.s2[ind]).to(self.device),
            torch.FloatTensor(self.a2[ind]).to(self.device),
            torch.FloatTensor(self.mask2[ind]).to(self.device),
            torch.FloatTensor(self.labels[ind]).to(self.device),
        )
