import numpy as np
import torch
import h5py

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
        config,
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
        self.subsample_size = config['subsample_size']

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

        self.iter = 0
        self.permutation = None
    
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
        index = np.load(self.pref_dir).astype(int)

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
        self.permutation = np.random.permutation(self.total_feedbacks)
    
    def sample(self, batch_size):
        iters = np.ceil(self.total_feedbacks / batch_size)
        ind = self.permutation[self.iter * batch_size : min((self.iter + 1) * batch_size, self.total_feedbacks)]   
        if self.iter == iters-1:
            self.iter == 0
            self.permutation = np.random.permutation(self.total_feedbacks)
        else:
            self.iter += 1   
        
        if self.subsample_size is None:
            return {
            'obs1': torch.FloatTensor(self.o1[ind]).to(self.device),
            'state1': torch.FloatTensor(self.s1[ind]).to(self.device),
            'action1': torch.FloatTensor(self.a1[ind]).to(self.device),
            'mask1': torch.FloatTensor(self.mask1[ind]).to(self.device),
            'obs2': torch.FloatTensor(self.o2[ind]).to(self.device),
            'state2': torch.FloatTensor(self.s2[ind]).to(self.device),
            'action2': torch.FloatTensor(self.a2[ind]).to(self.device),
            'mask2': torch.FloatTensor(self.mask2[ind]).to(self.device),
            'labels': torch.FloatTensor(self.labels[ind]).to(self.device),
            'r1': torch.FloatTensor(self.r1[ind]).to(self.device),
            'r2': torch.FloatTensor(self.r2[ind]).to(self.device),
            }
        else:
            # Note: subsample sequences currently do not support arbitrary obs/action spaces.
            start = np.random.randint(0, self.segment_length - self.subsample_size)
            end = start + self.subsample_size
            return {
            'obs1': torch.FloatTensor(self.o1[ind, start:end]).to(self.device),
            'state1': torch.FloatTensor(self.s1[ind, start:end]).to(self.device),
            'action1': torch.FloatTensor(self.a1[ind, start:end]).to(self.device),
            'mask1': torch.FloatTensor(self.mask1[ind, start:end]).to(self.device),
            'obs2': torch.FloatTensor(self.o2[ind, start:end]).to(self.device),
            'state2': torch.FloatTensor(self.s2[ind, start:end]).to(self.device),
            'action2': torch.FloatTensor(self.a2[ind, start:end]).to(self.device),
            'mask2': torch.FloatTensor(self.mask2[ind, start:end]).to(self.device),
            'labels': torch.FloatTensor(self.labels[ind]).to(self.device),
            'r1': torch.FloatTensor(self.r1[ind, start:end]).to(self.device),
            'r2': torch.FloatTensor(self.r2[ind, start:end]).to(self.device),
            }
