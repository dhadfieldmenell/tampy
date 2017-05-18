import unittest, os, h5py, time, scipy.stats, main
import matplotlib.pylab as plt
import numpy as np
import sys
sys.path.append(os.path.abspath("/home/simon0xzx/Research/tampy/src"))
from core.util_classes.learning import PostLearner
from core.parsing import parse_domain_config, parse_problem_config
from core.util_classes.can import GreenCan
from core.util_classes.viewer import OpenRAVEViewer
from core.util_classes.plan_hdf5_serialization import PlanDeserializer
from core.util_classes.openrave_body import OpenRAVEBody
from pma import hl_solver
from pma import robot_ll_solver

# @profile
def test_realistic_training():
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

    plans = []
    result = []
    #8 doesn't work
    plan_list = [1, 2, 3]
    for i in plan_list:
        print "Generating plan_{}".format(i)
        prob_file = '../domains/baxter_domain/baxter_training_probs/grasp_training_4321_{}.prob'.format(i)

        plan_str = ['0: MOVETO BAXTER ROBOT_INIT_POSE PDP_TARGET0',
                    '1: GRASP BAXTER CAN0 TARGET0 PDP_TARGET0 EE_TARGET0 PDP_TARGET1',
                    '2: MOVETOHOLDING BAXTER PDP_TARGET1 ROBOT_END_POSE CAN0']

        plan = get_plan(prob_file, plan_str)
        plans.append(plan)

        geom = plan.params['can0'].geom
        plan.params['can0'].geom = GreenCan(geom.radius, geom.height)



        solver = robot_ll_solver.RobotLLSolver()
        def viewer():
            return OpenRAVEViewer.create_viewer(plan.env)
        timesteps = solver.solve(plan, callback=viewer, n_resamples=5, verbose=False)
        result.append(plan.sampling_trace)
        # hdf5 = h5py.File("features{}.hdf5".format(i), "w")
        # f, r = trace_to_data(plan.sampling_trace)
        # arg_dict = {'train_size': 1, 'episode_size': 5, 'train_stepsize': 0.05, 'sample_iter': 1000, 'sample_burn': 250, 'sample_thin': 3}
        # learner = PostLearner(arg_dict, "test_learner")
        # param_dict = {'Robot':{'rArmPose':7},
        #               'EEPose':{'value':3},
        #               'RobotPose': {'rArmPose': 7}}
        # learner.train([f], [r], param_dict)

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


test_realistic_training()
