import numpy as np
import h5py

def make_pref_index(num_feedbacks, segment_length, file_dir, env_name):
    print('==========Data loading==========')
    data_file = file_dir + env_name + ".hdf5"
    # data_file = self.data_dir + 'test.hdf5'
    print('Loading from:', data_file)
    f = h5py.File(data_file, 'r')
    d = np.array(f['d'])
    f.close()

    data_size = d.shape[0]
    nonterminal_steps, = np.where(
        np.logical_and(
            np.logical_not(d[:,0]),
            np.arange(data_size) < data_size - 1))
    terminal_steps, = np.where(
        np.logical_and(d[:,0], np.arange(data_size) < data_size))
    print('Found %d non-terminal steps out of a total of %d steps.' % (
        len(nonterminal_steps), data_size))
    print('Found %d total episodes out of %d steps.' % (
        len(terminal_steps), data_size))
    
    # consider to add a total same transition in terminal step to avoid discontinuity
    episode_lens = np.insert(terminal_steps, 0, -1)[1:] - np.insert(terminal_steps, 0, -1)[:-1] - 1
    episode_index = np.where(episode_lens > segment_length)[0]
    sample_index1 = np.random.choice(episode_index.shape[0], size=num_feedbacks, replace=True)
    sample_index2 = np.random.choice(episode_index.shape[0], size=num_feedbacks, replace=True)

    # 2 refers to 2 sgemnets, 1 referd to end index
    index = np.zeros((num_feedbacks, 2))

    for i in range(num_feedbacks):
        index1, index2 = episode_index[sample_index1[i]], episode_index[sample_index2[i]]
        time_index_1 = np.random.choice(episode_lens[index1] - segment_length + 1)
        time_index_1 = terminal_steps[index1] - 1 - time_index_1
        time_index_2 = np.random.choice(episode_lens[index2] - segment_length + 1)
        time_index_2 = terminal_steps[index2] - 1 - time_index_2
        index[i][0], index[i][1] = int(time_index_1), int(time_index_2)
    
    np.save(file_dir + env_name + "-" + str(num_feedbacks) + "-" + str(segment_length) + "pref_index.npy", index)



if __name__ == "__main__":
    # TODO: currently use random sample, after add uncertainty sample methods
    file_dir = "/data/user/kouqian/files/MAOfflineDatasets/MA-Mujoco/"
    env_name = "HalfCheetah-v2-6x1-medium"
    num_feedbacks = 30000
    segment_length = 100
    make_pref_index(num_feedbacks, segment_length, file_dir, env_name)
