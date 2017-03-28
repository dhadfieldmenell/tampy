import unittest, os, h5py
import numpy as np
import scipy.stats
from core.util_classes.learning import PostLearner
from math import *
import main
from core.parsing import parse_domain_config, parse_problem_config
import matplotlib.pylab as plt
class TestLearner(unittest.TestCase):

    def test_initialize_file_storage(self):
        if os.path.isfile("test_learner.hdf5"):
            os.remove("test_learner.hdf5")
        arg_dict = {'train_size': 10, 'episode_size': 5, 'solver': None, 'train_stepsize': 0.05, 'sample_iter': 1000, 'sample_burn': 250, 'sample_thin': 3}
        learner = PostLearner(arg_dict, "test_learner", "CONFIG")
        self.assertEqual(learner.store_file, "test_learner.hdf5")
        self.assertEqual(learner.sample_space(), "CONFIG")
        self.assertEqual(learner.theta, None)
        self.assertFalse(learner.trained)
        data = {"robot":{"lArmPose": np.arange(7),
                         "rArmPose": np.arange(7),
                         "pose": np.zeros((1,)),
                         "lGripper": np.ones((1,)),
                         "rGripper": np.ones((1,))
                         },
               "can1": {"pose": np.arange(3),
                        "rotation": np.zeros((3,))
                       },
               "target1": {"value": np.arange(3),
                           "rotation": np.zeros((3,))
                          },
               "can1": {"pose": np.arange(3),
                        "rotation": np.zeros((3,))
                       },
               "target1": {"value": np.arange(3),
                           "rotation": np.zeros((3,))
                          }
               }
        learner.theta = data
        hdf5 = h5py.File("test_learner.hdf5", "w")
        group = hdf5.create_group("THETA_CONFIG")
        learner.store_theta(group)
        hdf5.close()
        hdf5 = h5py.File("test_learner.hdf5", "r")
        group = hdf5.values()[0]
        theta = learner.get_theta(group)
        for param in theta:
            attr_dict = theta[param]
            for attr in attr_dict:
                data_value = data[param][attr]
                theta_value = data[param][attr]
                self.assertEqual(np.allclose(data_value, theta_value), True)
        hdf5.close()
        learner2 = PostLearner(arg_dict, "test_learner", "CONFIG")
        self.assertTrue(learner2.theta != None)
        self.assertTrue(learner2.trained)
        os.remove("test_learner.hdf5")

    def test_metropolis_hasting1(self):
        # Testing with single variable Normal Distribution
        arg_dict = {'train_size': 10, 'episode_size': 5, 'solver': None, 'train_stepsize': 0.05, 'sample_iter': 50000, 'sample_burn': 500, 'sample_thin': 2}
        learn = PostLearner(arg_dict, "test_learner", "CONFIG")
        data = {"dummy":{"x": np.array([[1]])}}
        learn.theta = data
        def feature_fun(x):
            return x

        def norm_pdf(data, old):
            return scipy.stats.norm.pdf(data, old, 2)

        def sample_norm(old_alpha):
            return np.random.normal(old_alpha, 2, len(old_alpha))

        def sample_logistic(old_alpha):
            return np.random.logistic(old_alpha, 2, len(old_alpha))

        def logistic_pdf(data, old):
            return scipy.stats.logistic.pdf(data, old, 2)

        boundary = np.array([[-3, 3]])
        samples1 = learn.metropolis_hasting(boundary, feature_fun, lambda x:norm_pdf(x, 5), sample_logistic, logistic_pdf)

        x = np.arange(-5,15,.1)
        y = [norm_pdf(i, 5) for i in x]
        plt.subplot(211)
        plt.title('Metropolis-Hastings')
        plt.plot(samples1)
        plt.subplot(212)

        plt.hist(samples1[1000::3], bins=30,normed=1)
        plt.plot(x,y,'ro')
        plt.ylabel('Frequency')
        plt.xlabel('x')
        plt.legend(('PDF','Samples'))
        plt.show()

        samples2 = learn.metropolis_hasting(boundary, feature_fun, lambda x:logistic_pdf(x, 5), sample_norm, norm_pdf)

        x = np.arange(-5,15,.1)
        y = [logistic_pdf(i, 5) for i in x]
        plt.subplot(211)
        plt.title('Metropolis-Hastings')
        plt.plot(samples2)
        plt.subplot(212)

        plt.hist(samples2[1000::3], bins=30,normed=1)
        plt.plot(x,y,'ro')
        plt.ylabel('Frequency')
        plt.xlabel('x')
        plt.legend(('PDF','Samples'))
        plt.show()

    def test_training(self):
        arg_dict = {'train_size': 10, 'episode_size': 5, 'solver': None, 'train_stepsize': 0.05, 'sample_iter': 10000, 'sample_burn': 500, 'sample_thin': 2}
        learn = PostLearner(arg_dict, "test_learner", "CONFIG")

        # domain, problems = load_environment('../domains/baxter_domain/baxter.domain', '../domains/baxter_domain/baxter_training_probs/grasp_training_4321_', 20)
        # import ipdb; ipdb.set_trace()


    def test_sampling(self):
        pass




def load_environment(domain_file, problem_file, problem_size):
    domain_fname = domain_file
    d_c = main.parse_file_to_dict(domain_fname)
    domain = parse_domain_config.ParseDomainConfig.parse(d_c)
    problems = []
    for i in range(problem_size):
        print "prob_{}".format(i)
        p_fname = problem_file + '{}.prob'.format(i)
        p_c = main.parse_file_to_dict(p_fname)
        prob = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
        problems.append(prob)
    return domain, problems

if __name__ == "__main__":
    unittest.main()
