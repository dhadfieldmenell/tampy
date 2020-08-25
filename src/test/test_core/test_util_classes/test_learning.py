import unittest, os, h5py, time, main, scipy.stats
import matplotlib.pylab as plt
import numpy as np
from core.util_classes.learning import PostLearner
from core.parsing import parse_domain_config, parse_problem_config
from core.util_classes.can import GreenCan
from core.util_classes.viewer import OpenRAVEViewer
from core.util_classes.plan_hdf5_serialization import PlanDeserializer
from core.util_classes.openrave_body import OpenRAVEBody
from pma import hl_solver
from pma import robot_ll_solver

class TestLearner(unittest.TestCase):

    def test_initialize_file_storage(self):
        if os.path.isfile("test_learner.hdf5"):
            os.remove("test_learner.hdf5")
        arg_dict = {'train_size': 10, 'episode_size': 5, 'train_stepsize': 0.05, 'sample_iter': 1000, 'sample_burn': 250, 'sample_thin': 3}
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
        group = list(hdf5.values())[0]
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
        debug = False
        arg_dict = {'train_size': 10, 'episode_size': 5, 'train_stepsize': 0.05, 'sample_iter': 50000, 'sample_burn': 500, 'sample_thin': 2}
        learn = PostLearner(arg_dict, "test_learner", "CONFIG")
        data = {"dummy":{"x": np.array([[1]])}}
        learn.theta = data
        def feature_fun(x):
            return x

        def norm_pdf(data, old):
            return scipy.stats.norm.pdf(data, old, .04)

        def sample_norm(old_alpha):
            return np.random.normal(old_alpha, 0.08, len(old_alpha))

        def lognorm_pdf(data):
            return scipy.stats.lognorm.pdf(data, 1)

        def alpha(data):
            return scipy.stats.alpha.pdf(data, 0.5)

        def gumbel_r(data):
            return scipy.stats.gumbel_r.pdf(data)

        def logistic_pdf(data):
            return scipy.stats.logistic.pdf(data, 5, 2)

        def gibbs(data):
            return np.exp(learn.theta['dummy']['x'].T.dot(data))


        boundary = np.array([[-1, 1]])
        samples = learn.metropolis_hasting(boundary, feature_fun, gibbs, sample_norm, norm_pdf)
        x = np.arange(-1,1,.1)
        y = np.array([gibbs(i) for i in x])
        y = y.reshape((y.shape[0], ))
        plt.subplot(211)
        plt.title('Metropolis-Hastings')
        plt.plot(samples)
        plt.subplot(212)

        plt.hist(samples, bins=30,normed=1)
        plt.plot(x,y,'ro')
        plt.ylabel('Frequency')
        plt.xlabel('x')
        plt.legend(('PDF','Samples'))
        print("data sample generated successfully.")
        if debug == True:
            plt.show()

    def test_training(self):
        np.random.seed(1234)
        arg_dict = {'train_size': 2, 'episode_size': 2,  'train_stepsize': 0.05, 'sample_iter': 10000, 'sample_burn': 500, 'sample_thin': 2}
        learn = PostLearner(arg_dict, "test_learner", "CONFIG")
        param_dict = {'dummy': {'x': 1, 'y':2, 'z':3}}
        feature_vecs = [{'dummy': {'x': [np.array([8]), np.array([-7])],
                                   'y': [np.array([[6, 3]]).T, np.array([[ 7], [-6]])],
                                   'z': [np.array([[-3, 1, 2]]).T, np.array([[-4, -6, -1]]).T]}},
                        {'dummy': {'x': [np.array([[-6]]), np.array([[-3]])],
                                   'y': [np.array([[-7, -1]]).T, np.array([[7, 7]]).T],
                                   'z': [np.array([[-2,  8,  6]]).T, np.array([[5, 4, 8]]).T]}}]
        rewards =  [{'dummy': {'x': [4, 5],
                               'y': [-9,  7],
                               'z': [-6, -8]}},
                    {'dummy': {'x': [ 0, -4],
                               'y': [-4, -2],
                               'z': [-5,  4]}}]
        learn.train_config(feature_vecs, rewards, param_dict)
        self.assertTrue(np.allclose(learn.theta['dummy']['x'], np.array([[-3.22769282]])))
        self.assertTrue(np.allclose(learn.theta['dummy']['y'], np.array([[ 2.24185377, 2.26802808]]).T))
        self.assertTrue(np.allclose(learn.theta['dummy']['z'], np.array([[ 0.9911284, 3.3078589, 1.27975853]]).T))

    def test_sampling(self):

        debug = False

        # Testing with single variable Normal Distribution
        arg_dict = {'train_size': 10, 'episode_size': 5, 'solver': None, 'train_stepsize': 0.05, 'sample_iter': 10000, 'sample_burn': 250, 'sample_thin': 2}
        learn = PostLearner(arg_dict, "test_learner", "CONFIG")

        test_x = np.array([[-1]]).T
        test_feature = lambda x: np.array([[x[0]]]).T
        test_theta = np.array([[-1]]).T
        test_boundary = np.array([[0, 10]])

        learn.theta = {"dummy":{"x": test_theta}}
        feature_dict = {"dummy": {"x": test_feature}}
        param_dict = {"dummy": {"x": test_boundary}}

        sample_dict = learn.sample(param_dict, feature_dict)

        sample = sample_dict["dummy"]["x"]
        sample_pdf  = lambda alpha: learn.train_model(learn.theta["dummy"]["x"], feature_dict["dummy"]["x"](alpha))

        x = np.arange(0,10,.1)
        y = np.array([sample_pdf([i]) for i in x])
        y = y.reshape((y.shape[0], ))
        plt.subplot(211)
        plt.title('Metropolis-Hastings')
        plt.plot(sample)
        plt.subplot(212)
        plt.hist(sample, bins=30,normed=1)
        plt.plot(x,y,'ro')
        plt.ylabel('Frequency')
        plt.xlabel('x')
        plt.legend(('PDF','Samples'))
        print("data sample generated successfully.")
        if debug == True:
            plt.show()

    def test_realistic_training(self):
        domain_fname = '../domains/baxter_domain/baxter.domain'
        d_c = main.parse_file_to_dict(domain_fname)
        domain = parse_domain_config.ParseDomainConfig.parse(d_c)
        hls = hl_solver.FFSolver(d_c)
        def get_plan(p_fname, plan_str=None):
            p_c = main.parse_file_to_dict(p_fname)
            problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain)
            abs_problem = hls.translate_problem(problem)
            if plan_str is not None:
                return hls.get_plan(plan_str, domain, problem)
            return hls.solve(abs_problem, domain, problem)

        result = []
        #8 doesn't work
        plan_list = [1, 2, 3]
        for i in [1]:
            print("Generating plan_{}".format(i))
            prob_file = '../domains/baxter_domain/baxter_training_probs/grasp_training_4321_{}.prob'.format(i)

            plan_str = ['0: MOVETO BAXTER ROBOT_INIT_POSE PDP_TARGET0',
                        '1: GRASP BAXTER CAN0 TARGET0 PDP_TARGET0 EE_TARGET0 PDP_TARGET1',
                        '2: MOVETOHOLDING BAXTER PDP_TARGET1 ROBOT_END_POSE CAN0']

            plan = get_plan(prob_file, plan_str)

            geom = plan.params['can0'].geom
            plan.params['can0'].geom = GreenCan(geom.radius, geom.height)

            solver = robot_ll_solver.RobotLLSolver()
            def viewer():
                return OpenRAVEViewer.create_viewer(plan.env)
            timesteps = solver.solve(plan, callback=viewer, n_resamples=5, verbose=False)
            # result.append(plan.sampling_trace)
            # hdf5 = h5py.File("features{}.hdf5".format(i), "w")
            # f, r = trace_to_data(plan.sampling_trace)
            # arg_dict = {'train_size': 1, 'episode_size': 5, 'train_stepsize': 0.05, 'sample_iter': 1000, 'sample_burn': 250, 'sample_thin': 3}
            # learner = PostLearner(arg_dict, "test_learner")
            # param_dict = {'Robot':{'rArmPose':7},
            #               'EEPose':{'value':3},
            #               'RobotPose': {'rArmPose': 7}}
            # learner.train([f], [r], param_dict)
        # import ipdb; ipdb.set_trace()

def test_realistic_resampling(self):
    pass

def trace_to_data(sampling_trace):
    """
    Given a sampling trace, return a list of features and rewards
    """
    features, rewards = {}, {}
    for trace in sampling_trace:
        param = trace['type']
        if param not in features:
            features[param], rewards[param] = {}, {}
        for attr in trace['data']:
            if attr not in features[param]:
                features[param][attr] = []
                rewards[param][attr] = []
            features[param][attr].append(trace['data'][attr])
            rewards[param][attr].append(trace['reward'])

    return features, rewards

if __name__ == "__main__":
    unittest.main()
