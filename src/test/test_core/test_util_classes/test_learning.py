import unittest, os, h5py
import numpy as np
from core.util_classes.learning import PostLearner

class TestLearner(unittest.TestCase):

    def test_initialize(self):
        if os.path.isfile("test_learner.hdf5"):
            os.remove("test_learner.hdf5")
        arg_dict = {'train_size': 10, 'episode_size': 5, 'solver': None, 'sample_iter': 1000, 'sample_burn': 250, 'sample_thin': 3}
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

    def test_sampling(self):
        arg_dict = {'train_size': 10, 'episode_size': 5, 'solver': None, 'sample_iter': 1000, 'sample_burn': 250, 'sample_thin': 3}
        learn = PostLearner(arg_dict, "test_learner", "CONFIG")
        data = {"robot":{"lArmPose": np.ones((7,)),
                         "rArmPose": np.ones((7,)),
                         "pose": np.ones((1,)),
                         "lGripper": np.ones((1,)),
                         "rGripper": np.ones((1,))
                         }
               }
        learn.theta = data
        def feature_fun(x):
            return x
        param_dict = {"robot":{"lArmPose": np.hstack([np.zeros((7,1)), 10*np.ones((7,1))]),
                         "rArmPose": np.hstack([np.zeros((7,1)), 10*np.ones((7,1))]),
                         "pose": np.array([[0, 10]]),
                         "lGripper": np.array([[0, 10]]),
                         "rGripper": np.array([[0, 10]])
                         }
               }
        sample = learn.sample_config(None, param_dict, feature_fun)
        import ipdb; ipdb.set_trace()




if __name__ == "__main__":
    unittest.main()
