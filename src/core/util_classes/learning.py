import h5py
import os.path as path
import numpy as np
import scipy.stats
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

    arg_dict_template = {'train_size': "int",
                         'episode_size': "int",
                         'solver': "LLSolver",
                         'train_stepsize': "float",
                         'sample_iter': "int",
                         'sample_burn': "int",
                         'sample_thin': "int"}

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
            print("File not found, please train this Learner.")

        self.train_size = arg_dict['train_size']
        self.episode_size = arg_dict['episode_size']
        self.sample_iter = arg_dict['sample_iter']
        self.sample_burn = arg_dict['sample_burn']
        self.sample_thin = arg_dict['sample_thin']
        self.train_stepsize = arg_dict['train_stepsize']

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

    def sample_space(self):
        return self._sample_space

    def norm_pdf(self, data, old):
        # If this function doesn't work upgrade your scipy version to SciPy 0.14.0.dev-16fc0af
        return scipy.stats.multivariate_normal.pdf(data, old, np.eye(len(data)))

    def sample_norm(self, old_alpha):
        return np.random.normal(old_alpha, .2, len(old_alpha))

    def train_model(self, theta, alpha):
        return np.exp(theta.T.dot(alpha))

    def train(self, feature_vecs, rewards, param_dict):
        if self._sample_space == "CONFIG":
            self.train_config(feature_vecs, rewards, param_dict)
        else:
            assert "Invalid Sample Space Specified"

    def train_config(self, feature_vecs, rewards, param_dict):
        """
            Args:
            feature_vecs: a list of N Dict mapping param/attr name to episode features where N is the number of problem trained.
                Type: List(Dict: param/attr -> value)
            rewards: a list of N Dict mapping param/attr name to reward of each episode, where N is the number of problem trained.
                Type: List(Dict: param/attr -> value)
            param_dict: A dict mapping param/attr name to dimension of theta wanted.
        """
        # Initialize theta variable
        self.theta = {}
        for param in param_dict:
            attr_dict = param_dict[param]
            self.theta[param] = {}
            for attr in attr_dict:
                rand = np.random.sample(attr_dict[attr])
                self.theta[param][attr] = rand.reshape((rand.shape[0], 1))

        for i in range(self.train_size):
            features = feature_vecs[i]
            reward = rewards[i]
            for param in param_dict:
                attr_dict = param_dict[param]
                for attr in attr_dict:
                    if param not in features or attr not in features[param]:
                        continue
                    feature_list = features[param][attr]
                    reward_list = reward[param][attr]
                    dist_list = np.array([np.exp(self.theta[param][attr].T.dot(feature)) for feature in feature_list]).reshape((len(feature_list),))
                    # TODO this sum might not necessarily representing Z
                    dist = dist_list/float(np.sum(dist_list))

                    expected = 0
                    for j in range(len(feature_list)):
                        expected += feature_list[j] * dist[j]

                    grad = 0
                    for k in range(len(feature_list)):
                        grad += feature_list[k] - expected
                    gradient = np.sum(reward_list) / float(len(feature_list)) * grad

                    self.theta[param][attr] = self.theta[param][attr] + self.train_stepsize * gradient
        f = h5py.File(self.store_file, 'w')
        self.store_theta(f)
        f.close()

    def sample(self, param_dict, feature_dict):
        if self._sample_space == "CONFIG":
            return self.sample_config(param_dict, feature_dict)
        else:
            assert "Invalid Sample Space Specified"

    def sample_config(self, param_dict, feature_dict):
        """
        sample according to Giab Distribution biased by parameter self.theta.
        Return dictionary mapping param_name to sampled value.

        Args:
        feature_dict: A dict mapping param/attr name to feature function
        (dict: param_name->(attr_dict: attr->value))
        param_dict: A dict mapping param/attr name to value upper and lower bound of this parameter value
        """

        sample_dict = {}
        for param in param_dict:
            attr_dict = param_dict[param]
            sample_dict[param] = {}
            for attr in attr_dict:
                def sample_pdf(data):
                    return self.train_model(self.theta[param][attr], feature_dict[param][attr](data))

                result_sampling = self.metropolis_hasting(attr_dict[attr], feature_dict[param][attr], sample_pdf, self.sample_norm, self.norm_pdf)
                sample_dict[param][attr] = result_sampling
                # Probably needs some filtering to get rid of parameter value that are out of bound
        return sample_dict

    def metropolis_hasting(self, boundary, feature_fun, model, sample_step, prop_pdf):
        """
            This function implements Metropolis Hasting Algorithm for sampling.
            Arg:
                Boundary: A kx2 matrix defines lower bound(first column) and upper bound(last column) of the sample you wish to draw.

                Feature_fun: Defines a mapping between sample and it's coresponding feature vector.

                Model: is pdf from which the sample is draw from

                Sample_step: A proposal distribution to draw sample from.
                prop_pdf: PDF of proposal distribution
        """
        # Give a reasonable initial guess for alpha
        old_alpha = np.multiply(np.random.sample(boundary.shape[0]), boundary[:,1] - boundary[:,0]) + boundary[:,0]
        samples = [old_alpha]
        # Initialize x' as zero
        new_alpha = np.zeros((len(old_alpha),))
        # Metropolis-Hastings with 10,000 iterations.
        for n in range(self.sample_iter):
            # New sample x' is drawn from the proposal distribution g(x'|x)
            new_alpha = sample_step(old_alpha)
            # Constraint x within the boundary
            for i in range(new_alpha.shape[0]):
                if new_alpha[i] < boundary[i, 0]:
                    # If data entry is smaller than lower bound
                    new_alpha[i] = boundary[i, 0]
                elif new_alpha[i] > boundary[i, 1]:
                    # If data entry is larger than upper bound
                    new_alpha[i] = boundary[i, 1]

            # Old Likelihood = P(x)g(x'|x)
            likelihood = model(old_alpha) * prop_pdf(new_alpha, old_alpha)
            # New Likelihood = P(x')g(x|x')
            new_likelihood = model(new_alpha) * prop_pdf(old_alpha, new_alpha)
            # A(x'|x) = min(1, (P(x')g(x|x'))/(P(x)g(x'|x)))
            accept_prob = min([1., new_likelihood / float(likelihood)])
            u = random.uniform(0.0,1.0)
            # Accept new sample according to accpetance probability
            if u < accept_prob:
                old_alpha = new_alpha
                samples.append(new_alpha)

        # remove burn in period and skip samples to reduce autocorrelation
        return np.array(samples[self.sample_burn::self.sample_thin])
