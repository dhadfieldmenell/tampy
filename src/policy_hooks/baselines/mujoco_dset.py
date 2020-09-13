'''
Data structure of the input .npz:
the data is save in python dictionary format with keys: 'acs', 'ep_rets', 'rews', 'obs'
the values of each item is a list storing the expert trajectory sequentially
a transition can be: (data['obs'][t], data['acs'][t], data['obs'][t+1]) and get reward data['rews'][t]
'''

from baselines import logger
import numpy as np


class Dset(object):
    def __init__(self, inputs, labels, randomize):
        self.inputs = inputs
        self.labels = labels
        assert len(self.inputs) == len(self.labels)
        self.randomize = randomize
        self.num_pairs = len(inputs)
        self.init_pointer()

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx, :]
            self.labels = self.labels[idx, :]

    def get_next_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs, self.labels
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end, :]
        labels = self.labels[self.pointer:end, :]
        self.pointer = end
        return inputs, labels


class Mujoco_Dset(object):
    def __init__(self, expert_path, gen_f, train_fraction=0.7, traj_limitation=-1, randomize=True):
        if type(expert_path) is str:
            expert_path = [expert_path]
        self.gen_f = gen_f
        traj_data = np.load(expert_path.pop(), allow_pickle=True)
        traj_data = traj_data[()]
        while len(expert_path) and (traj_limitation < 0 or len(traj_data['obs']) < traj_limitation):
            next_f = expert_path.pop()
            try:
                next_data = np.load(next_f, allow_pickle=True)
                next_data = next_data[()]
                for key in traj_data:
                    traj_data[key].extend(next_data[key])
            except Exception as e:
                logger.log('Could not load pickled dataset from {0}'.format(next_f))
        if traj_limitation < 0:
            traj_limitation = len(traj_data['obs'])
        obs = np.array(traj_data['obs'][:traj_limitation])
        acs = np.array(traj_data['acs'][:traj_limitation])

        # obs, acs: shape (N, L, ) + S where N = # episodes, L = episode length
        # and S is the environment observation/action space.
        # Flatten to (N * L, prod(S))
        if len(obs.shape) > 2:
            self.obs = np.reshape(obs, [-1, np.prod(obs.shape[2:])])
            self.acs = np.reshape(acs, [-1, np.prod(acs.shape[2:])])
        else:
            self.obs = np.vstack(obs)
            self.acs = np.vstack(acs)

        self.rets = traj_data['ep_rets'][:traj_limitation]
        self.avg_ret = np.sum([np.sum(r) for r in self.rets])/len(self.rets)
        self.std_ret = np.std(np.array([np.sum(r) for r in self.rets]))
        if len(self.acs) > 2:
            self.acs = np.squeeze(self.acs)
        assert len(self.obs) == len(self.acs)
        self.num_traj = min(traj_limitation, len(traj_data['obs']))
        self.num_transition = len(self.obs)
        self.randomize = randomize
        self.dset = Dset(self.obs, self.acs, self.randomize)
        # for behavior cloning
        self.train_set = Dset(self.obs[:int(self.num_transition*train_fraction), :],
                              self.acs[:int(self.num_transition*train_fraction), :],
                              self.randomize)
        self.val_set = Dset(self.obs[int(self.num_transition*train_fraction):, :],
                            self.acs[int(self.num_transition*train_fraction):, :],
                            self.randomize)
        self.log_info()

    def log_info(self):
        logger.log("Total trajectories: %d" % self.num_traj)
        logger.log("Total transitions: %d" % self.num_transition)
        logger.log("Average returns: %f" % self.avg_ret)
        logger.log("Std for returns: %f" % self.std_ret)

    def get_next_batch(self, batch_size, split=None):
        if split is None:
            return self.dset.get_next_batch(batch_size)
        elif split == 'train':
            return self.train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError

    def plot(self):
        import matplotlib.pyplot as plt
        plt.hist(self.rets)
        plt.savefig("histogram_rets.png")
        plt.close()


def test(expert_path, traj_limitation, plot):
    dset = Mujoco_Dset(expert_path, traj_limitation=traj_limitation)
    if plot:
        dset.plot()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_path", type=str, default="../data/deterministic.trpo.Hopper.0.00.npz")
    parser.add_argument("--traj_limitation", type=int, default=None)
    parser.add_argument("--plot", type=bool, default=False)
    args = parser.parse_args()
    test(args.expert_path, args.traj_limitation, args.plot)
