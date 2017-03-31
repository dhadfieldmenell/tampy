import h5py
import os.path as path
from IPython import embed as shell
import numpy as np

class Learner(object):
    """
    Base class for all learner.
    """
    def __init__(self, file_name):
        self.theta = None
        self.trained = False
        self.store_file = file_name
        if self.store_file[-5:] != '.hdf5':
            self.store_file += '.hdf5'

    def train(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

class PostLearner(Learner):

    def __init__(self, file_name, train_size, episode_size, space = "CONFIG"):
        """
        Constructor will first try to load the feature value theta
        If no useful information is loaded, we will mark this learner
        as untrained.
        """
        super(PostLearner, self).__init__(file_name)
        if space == "CONFIG" or space == "WORK":
            self._sample_space = space
        else:
            self._sample_space = "UNDEFINED"
            assert "Invalid sample space. sapce has to be either CONFIG for configuration space or WORK for work space"

        try:
            f = h5py.File(self.store_file, 'r')
            if "/THETA_" + self._sample_space in f:
                self.trained = True
                self.theta = self.get_theta(f["/THETA_" + self._sample_space])
            f.close()
        except:
            print "File not found, please train this Learner."

        self.train_size = train_size
        self.episode_size = episode_size

    def sample_space(self):
        return self._sample_space

    def get_theta(self, hdf5_obj):
        """
            Recursively load theta value from hfd5 oject into a dictionary
            hdf5 structure:
            hdf5_obj->pred_group->param_group->attr_dataset
        """
        theta = {}
        for param in hdf5_obj:
            param_obj = hdf5_obj[param]
            param_dict = {}
            for attr in param_obj:
                param_dict[attr] = param_obj[attr].value
            theta[param] = param_dict
        return theta

    def store_theta(self, hdf5_obj):
        """
            Recursively store theta value from self.theta into hdf5 object
            hdf5 structure:
            hdf5_obj->param_group->attr_dataset
        """
        for param in self.theta:
            param_group = hdf5_obj.create_group(param)
            attr_dict = self.theta[param]
            for attr in attr_dict:
                attr_theta = param_group.create_dataset(attr, attr_dict[attr].shape)
                attr_theta[...] = attr_dict[attr]


    def change_space(self, new_space):
        """
        This function allows user to switch space to sample from.
        of course switching the sapce requires resampling
        """
        if new_space == self._sample_space:
            return
        elif new_space == "CONFIG" or new_space == "WORK":
            self._sample_space = new_space
            self.trained = False
            self.theta = None
            f = h5py.File(self.store_file, 'r')
            if "/THETA_" + self._sample_space in f:
                self.trained = True
                self.theta = f["/THETA_" + self._sample_space].value
            f.close()
        else:
            self._sample_space = "UNDEFINED"
            assert "Invalid sample space. new sapce has to be either CONFIG for configuration space or WORK for work space"

    def train(self, domain, problem, feature_fun):
        if self._sample_space == "CONFIG":
            self.train_config(domain, problem, feature_fun)
        elif self._sample_space == "WORK":
            self.train_work(domain, problem, feature_fun)
        else:
            assert "Invalid Sample Space Specified"

    def train_work(self, domain, problem, feature_fun):
        pass

    def train_config(self, domain, problem, feature_fun):
        pass

    def sample(self, state, feature_fun):
        if self._sample_space == "CONFIG":
            self.sample_config(state, feature_fun)
        elif self._sample_space == "WORK":
            self.sample_work(state, feature_fun)
        else:
            assert "Invalid Sample Space Specified"

    def sample_work(self, state, feature_fun):
        pass

    def sample_config(self, state, feature_fun):
        """
        Given a feature_mapping:
        (dict: param_name->(attr_dict: attr->value))
        sample according to Giab Distribution biased by parameter self.theta.
        Return dictionary mapping param_name to sampled value.
        """
        #TODO Currently not exactly correct
        config_value = {}
        for param in self.theta:
            attr_dict = self.theta[param]
            for attr in attr_dict:
                feature = feature_fun[param][attr]
                theta = self.theta[param][attr]
                config_value[param][attr] = self.sample_helper(theta, feature)
        return config_value

    def sample_helper(self, theta, feature):
        sample_list = []
        for i in range(self.episode_size):
            sample_list.append(np.exp(theta.dot(feature)))
        sample_list = np.array(sample_list) / np.linalg.norm(sample_list)
        accumulate = [0]
        for sample in sample_list:
            accumulate.append(accumulate[-1] + sample)
        beta = np.random.uniform()
        #TODO not yet finished#
