import h5py
import os.path as path
from IPython import embed as shell
import numpy as np
import math, random

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
    """
    User defined element:
    A distribution over the plan,
    feature functions for for the parameters,
    what parameters are needed to be trained.
    """

    def __init__(self, arg_dict, file_name, space = "CONFIG"):
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

        self.train_size = arg_dict['train_size']
        self.episode_size = arg_dict['episode_size']
        self.solver = arg_dict['solver']
        self.sample_iter = arg_dict['sample_iter']
        self.sample_burn = arg_dict['sample_burn']
        self.sample_thin = arg_dict['sample_thin']
        self.train_stepsize = arg_dict['train_stepsize']

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

    def train(self, problem, feature_fun, param_dict):
        if self._sample_space == "CONFIG":
            self.train_config(problem, feature_fun, param_dict)
        elif self._sample_space == "WORK":
            self.train_work(problem, feature_fun, param_dict)
        else:
            assert "Invalid Sample Space Specified"

    def train_work(self, problem, feature_fun, param_dict):
        pass

    def train_config(self, problems, feature_fun, param_dict):
        """
            Solver needs to provide a method train_sample
            that takes in a problem, episode_size, and feature_fun
            and returns each sample episode's feature vector (List(Dict:param->feature))
            and its reward (List)
        """

        for prob in problems:
            features, reward = solver.train_sample(porb, self.episode_size, feature_fun)
            R = np.sum(reward)
            for param in param_dict:
                for attr in param:
                    dist_list = np.array([np.exp(self.theta[param][attr].dot(fea[param][attr])) for fea in features])
                    dist = dist_list/np.sum(dist_list)

                    expected = 0
                    for index in range(self.episode_size):
                        expected += features[i][param][attr]*dist[i]

                    grad = 0
                    for i in range(self.episode_size):
                        grad += features[i][param][attr] - expected
                    gradient = R / float(self.episode_size) * grad
                    self.theta[param][attr] = self.theta[param][attr] + self.train_stepsize* gradient
                    import ipdb; ipdb.set_trace()

    def sample(self, param_dict, feature_fun):
        if self._sample_space == "CONFIG":
            self.sample_config(param_dict, feature_fun)
        elif self._sample_space == "WORK":
            self.sample_work(param_dict, feature_fun)
        else:
            assert "Invalid Sample Space Specified"

    def sample_work(self, param_dict, feature_fun):
        pass

    def sample_config(self, param_dict, feature_fun):
        """
        Given a feature_mapping:
        (dict: param_name->(attr_dict: attr->value))
        sample according to Giab Distribution biased by parameter self.theta.
        Return dictionary mapping param_name to sampled value.
        """

        for param in param_dict:
            sample_dict[param] = {}
            for attr in param_dict[param]:
                def model(alpha):
                    return np.exp(self.theta[param][attr].T.dot(feature_fun(alpha)))

                def sample_step(prev):
                    return np.random.normal(0, 1, prev)

                result_sampling =  self.metropolis_hasting(param_dict[param][attr], feature_fun, model, sample_step)
                index = np.random.randint(0, len(result_sampling))
                sample_dict[param][attr] = result_sampling[index]
        return sample_dict

    def metropolis_hasting(self, boundary, feature_fun, model, sample_step):
        """
            This function implements Metropolis Hasting Algorithm for sampling.
            Arg:
                Boundary: A kx2 matrix defines lower bound(first column) and upper bound(last column) of the sample you wish to draw.
                Feature_fun: Defines a mapping between sample and it's coresponding feature vector.
                Model: is pdf from which the sample is draw from
                Sample_step: A function that takes in previous data point and sample the next.(should draw sample from the same distribution as Model)
        """
        # initial guess for alpha as array.
        old_alpha = np.multiply(np.random.sample(boundary.shape[0]), boundary[:,1] - boundary[:,0]) + boundary[:,0]
        samples = [old_alpha]
        new_alpha = np.zeros((len(old_alpha),))
        # Metropolis-Hastings with 10,000 iterations.
        for n in range(self.sample_iter):
            likelihood = model(old_alpha)
            # This new sample should be draw from the same distribution as model
            new_alpha = sample_step(old_alpha)
            new_likelihood = model(new_alpha)
            # Accept new candidate in Monte-Carlo fashing.
            accept_prob = min([1.,new_likelihood/float(likelihood)])
            u = random.uniform(0.0,1.0)
            if u < accept_prob:
                old_alpha = new_alpha
                samples.append(new_alpha)
        return np.array(samples[self.sample_burn::self.sample_thin])
