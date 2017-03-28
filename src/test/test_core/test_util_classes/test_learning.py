import unittest, os, h5py
import numpy as np
from core.util_classes.learning import PostLearner
from math import *
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

    def test_metropolis_hasting(self):
        # Testing with single variable Normal Distribution
        arg_dict = {'train_size': 10, 'episode_size': 5, 'solver': None, 'train_stepsize': 0.05, 'sample_iter': 50000, 'sample_burn': 500, 'sample_thin': 2}
        learn = PostLearner(arg_dict, "test_learner", "CONFIG")
        data = {"dummy":{"x": np.array([[1]])}}
        learn.theta = data
        def feature_fun(x):
            return x
        def sdnorm(z):
            """
            Standard normal pdf
            """
            return np.exp(-(z-5)*(z-5)/2*(1)**2)/sqrt(2*pi*(1)**2)

        def sample_step(old_alpha):
            return np.random.logistic(old_alpha, 3, len(old_alpha))

        boundary = np.array([[-3, 3]])
        samples = learn.metropolis_hasting(boundary, feature_fun, sdnorm, sample_step)

        x = np.arange(2,8,.1)
        y = [sdnorm(i) for i in x]
        plt.subplot(211)
        plt.title('Metropolis-Hastings')
        plt.plot(samples)
        plt.subplot(212)

        plt.hist(samples[1000::3], bins=30,normed=1)
        plt.plot(x,y,'ro')
        plt.ylabel('Frequency')
        plt.xlabel('x')
        plt.legend(('PDF','Samples'))
        plt.show()

    def test_training(self):
        arg_dict = {'train_size': 10, 'episode_size': 5, 'solver': None, 'train_stepsize': 0.05, 'sample_iter': 10000, 'sample_burn': 500, 'sample_thin': 2}
        learn = PostLearner(arg_dict, "test_learner", "CONFIG")
        data = {"dummy":{"x": np.array([[1]])}}
        learn.theta = data


    def test_sampling(self):
        pass






if __name__ == "__main__":
    unittest.main()
