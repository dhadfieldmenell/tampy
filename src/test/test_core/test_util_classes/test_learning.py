import unittest, os, h5py
import numpy as np
from core.util_classes.learning import PostLearner

class TestLearner(unittest.TestCase):

    def test_initialize(self):
        if os.path.isfile("test_learner.hdf5"):
            os.remove("test_learner.hdf5")
        learner = PostLearner("test_learner", "CONFIG")
        self.assertEqual(learner.store_file, "test_learner.hdf5")
        self.assertEqual(learner.sample_space(), "CONFIG")
        self.assertEqual(learner.theta, None)
        self.assertFalse(learner.trained)
        data = {"pred1": {"robot":{"lArmPose": np.arange(7),
                                   "rArmPose": np.arange(7),
                                   "pose": np.zeros((1,)),
                                   "lGripper": np.ones((1,)),
                                   "rGripper": np.ones((1,)) },
                          "can1": {"pose": np.arange(3),
                                   "rotation": np.zeros((3,))
                                  },
                          "target1": {"value": np.arange(3),
                                      "rotation": np.zeros((3,))
                                     }
                          },
                 "pred2": {"can1": {"pose": np.arange(3),
                                    "rotation": np.zeros((3,))
                                   },
                           "target1": {"value": np.arange(3),
                                       "rotation": np.zeros((3,))
                                      }
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
        for pred in theta:
            param_dict = theta[pred]
            for param in param_dict:
                attr_dict = param_dict[param]
                for attr in attr_dict:
                    data_value = data[pred][param][attr]
                    theta_value = theta_value = data[pred][param][attr]
                    self.assertEqual(np.allclose(data_value, theta_value), True)
        hdf5.close()
        learner2 = PostLearner("test_learner", "CONFIG")
        self.assertTrue(learner2.theta != None)
        self.assertTrue(learner2.trained)
        os.remove("test_learner.hdf5")

if __name__ == "__main__":
    unittest.main()
