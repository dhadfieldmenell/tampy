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

    def train_config(self, problems, feature_fun, param_dict, step_size = 0.05):
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
                    dist_list = np.array([np.exp(self.theta[param][attr].dot(fea[param][attr])) \
                                                            for fea in features])
                    dist = dist_list/np.sum(dist_list)

                    expected = 0
                    for index in range(self.episode_size):
                        expected += features[i][param][attr]*dist[i]

                    grad = 0
                    for i in range(self.episode_size):
                        grad += features[i][param][attr] - expected
                    gradient = R / float(self.episode_size) * grad
                    self.theta[param][attr] = self.theta[param][attr] + step_size* gradient

    def sample(self, state, param_dict, feature_fun):
        if self._sample_space == "CONFIG":
            self.sample_config(state, feature_fun)
        elif self._sample_space == "WORK":
            self.sample_work(state, feature_fun)
        else:
            assert "Invalid Sample Space Specified"

    def sample_work(self, state, param_dict, feature_fun):
        pass

    def sample_config(self, state, param_dict, feature_fun):
        """
        Given a feature_mapping:
        (dict: param_name->(attr_dict: attr->value))
        sample according to Giab Distribution biased by parameter self.theta.
        Return dictionary mapping param_name to sampled value.
        """
        def model(param, attr, alpha):
            np.exp(self.theta[param][attr].T.dot(feature_fun(alpha)))

        return self.metropolis_hasting(param_dict, feature_fun, model, self.sample_iter, self.sample_burn, self.sample_thin)

    def metropolis_hasting(self, param_dict, feature_fun, model, iteration, burn, thin):
        sample_dict = {}
        for param in param_dict:
            for attr in param_dict[param]:
                # initial guess for alpha as array.
                old_alpha = np.multiply(np.random.sample(param_dict[param][attr].shape[0]), param_dict[param][attr][:,1] - param_dict[param][attr][:,0]) + param_dict[param][attr][:,0]
                samples = [old_alpha]
                new_alpha = np.zeros((len(old_alpha),))
                # define stepsize of MCMC.
                stepsizes = [0.005] * len(old_alpha)  # array of stepsizes
                accepted  = 0.0
                # Metropolis-Hastings with 10,000 iterations.
                for n in range(iteration):
                    likelihood = model(param, attr, old_alpha)
                    accumulated_sum = likelihood
                    for i in range(len(old_alpha)):
                        # Use stepsize provided for every dimension.
                        new_alpha[i] = random.gauss(old_alpha[i], stepsizes[i])
                    # Suggest new candidate from Gaussian proposal distribution.
                    likelihood = model(param, attr, new_alpha)

                    # Accept new candidate in Monte-Carlo fashing.
                    accept_prob = min([1.,new_likelihood/likelihood])
                    u = random.uniform(0.0,1.0)
                    if u < aprob:
                        old_alpha = new_alpha
                        samples.append(new_alpha)
                result_sampling = samples[burn::thin]
                index = np.random.randint(0, len(result_sampling), 1)
                sample_dict[param][attr] = result_sampling[index]
