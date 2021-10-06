import numpy as np
import pybullet as p
import random

import robodesk

import main
from core.parsing import parse_domain_config, parse_problem_config
import core.util_classes.common_constants as const
from core.util_classes.robots import Baxter
from core.util_classes.openrave_body import *
from core.util_classes.transform_utils import *
from pma.hl_solver import *
from pma.pr_graph import *
from pma import backtrack_ll_solver as bt_ll
from pma.robot_solver import RobotSolver
import core.util_classes.transform_utils as T

from policy_hooks.multiprocess_main import load_config, setup_dirs, DIR_KEY
from policy_hooks.run_training import argsparser
from policy_hooks.utils.load_agent import *
import policy_hooks.robodesk.hyp as hyp
import policy_hooks.robodesk.desk_prob as prob

args = argsparser()
args.config = 'policy_hooks.robodesk.hyp'
args.render = True

base_config = hyp.refresh_config()
base_config['id'] = 0
base_config.update(vars(args))
base_config['args'] = args
config, config_module = load_config(args, base_config)
config.update(base_config)
config['load_render'] = True
agent_config = load_agent(config)
agent = build_agent(agent_config)
env = agent.base_env
mjc_env = agent.mjc_env


try:
    p.disconnect()
except Exception as e:
    print(e)

#const.NEAR_GRIP_COEFF = 5e-2
#const.GRASP_DIST = 0.2
#const.APPROACH_DIST = 0.025
#const.EEREACHABLE_ROT_COEFF = 8e-3
bt_ll.DEBUG = True
openrave_bodies = None
domain_fname = "../domains/robot_domain/right_desk.domain"
prob = "../domains/robot_domain/probs/robodesk_prob.prob"
d_c = main.parse_file_to_dict(domain_fname)
domain = parse_domain_config.ParseDomainConfig.parse(d_c)
hls = FFSolver(d_c)
p_c = main.parse_file_to_dict(prob)
visual = len(os.environ.get('DISPLAY', '')) > 0
#visual = False
problem = parse_problem_config.ParseProblemConfig.parse(p_c, domain, None, use_tf=True, sess=None, visual=visual)
params = problem.init_state.params


sucs = []
N_RUNS = 50
for run_num in range(N_RUNS):
    agent.mjc_env.reset()
    for param in ['ball', 'upright_block', 'flat_block', \
                  'drawer_handle', 'shelf_handle', 'shelf', 'drawer']:
        pose = agent.mjc_env.get_item_pose(param, euler=True)
        targ = '{}_init_target'.format(param)
        params[param].pose[:,0] = pose[0]
        params[param].rotation[:,0] = pose[1]
        if targ in params:
            params[targ].value[:,0] = pose[0]
            params[targ].rotation[:,0] = pose[1]
        #params[param].pose[:,0] = env.physics.named.data.qpos[param][:3]
        #quat = env.physics.named.data.qpos[param][3:7]
        #quat = [quat[1], quat[2], quat[3], quat[0]]
        #euler = T.quaternion_to_euler(quat)
        #params[param].rotation[:,0] = euler
    #params['ball'].rotation[:,0] = [0., -0.4, 1.57]
    params['drawer'].hinge[:,0] = agent.mjc_env.get_attr('drawer', 'hinge')
    params['shelf'].hinge[:,0] = agent.mjc_env.get_attr('shelf', 'hinge')

    right_jnts = agent.mjc_env.get_attr('panda', 'right')
    lb, ub = params['panda'].geom.get_joint_limits('right')
    lb = np.array(lb) + 2e-3
    ub = np.array(ub) - 2e-3
    right_jnts = np.clip(right_jnts, lb, ub)
    params['panda'].right[:,0] = right_jnts 

    params['robot_init_pose'].right[:,0] = agent.mjc_env.get_attr('panda', 'right')
    params['robot_init_pose'].right_gripper[:,0] = agent.mjc_env.get_attr('panda', 'right_gripper')

    for param in params:
        targ = '{}_init_target'.format(param)
        if targ in params:
            params[targ].value[:,0] = params[param].pose[:,0]
            params[targ].rotation[:,0] = params[param].rotation[:,0]

    #goal = '(and (InSlideDoor ball drawer) (Stacked upright_block flat_block) (NearGripperRight panda green_button))'
    goal = '(and (SlideDoorClose shelf_handle shelf) (Stacked upright_block flat_block) (NearGripperRight panda green_button))'
    #goal = '(and (SlideDoorOpen drawer_handle drawer) (NearApproachRight panda upright_block))'
    #goal = '(and (InSlideDoor upright_block shelf) (NearApproachRight panda ball))'
    #goal = '(and (SlideDoorClose drawer_handle drawer) (InSlideDoor ball drawer))'
    #goal = '(and (InSlideDoor ball drawer) (InSlideDoor upright_block shelf) (SlideDoorClose drawer_handle drawer) (SlideDoorClose shelf_handle shelf))'
    #goal = '(Stacked upright_block flat_block)'
    #goal = '(and (SlideDoorClose shelf_handle shelf) (InSlideDoor upright_block shelf))'
    #goal = '(Lifted flat_block panda)'
    #goal = '(Lifted upright_block panda)'
    goal = '(Lifted ball panda)'
    #goal = '(SlideDoorClose shelf_handle shelf)'
    #goal = '(SlideDoorOpen drawer_handle drawer)'
    #goal = '(InSlideDoor ball drawer)'
    goal = '(At flat_block bin_target)'
    #goal = '(At ball bin_target)'
    #goal = '(InSlideDoor upright_block shelf)'
    #goal = '(InSlideDoor flat_block shelf)'
    #goal = '(and (At flat_block bin_target) (At upright_block off_desk_target))'
    #goal = '(and (InSlideDoor ball drawer) (NearGripperRight panda green_button))'
    #goal = '(NearGripperRight panda green_button)'
    #goal = '(Lifted upright_block panda)'
    #goal = "(NearGripperRight panda green_button)"

    goals = [
         #'(Lifted upright_block panda)',
         #'(Lifted ball panda)',
         #'(Stacked upright_block flat_block)',
         #'(SlideDoorClose shelf_handle shelf)',
         #'(SlideDoorOpen drawer_handle drawer)',
         '(Near flat_block bin_target)',
         #'(Near upright_block off_desk_target)',
         #'(InSlideDoor flat_block shelf)',
         #'(InGripperRight panda green_button)',
            ]

    goal = random.choice(goals)
    goal_info = []
    goal_str = goal.strip()[1:-1]
    if goal_str.startswith('and'):
        goals = goal_str.split('(')
        for g in goals:
            if g.find(')') >= 0:
                goal_info.append(g.strip()[:-1].split(' '))
    else:
        goal_info = [goal_str.split(' ')]
    print('SOLVING:', goal, goal_info)

    print('CONSISTENT?', problem.init_state.is_consistent())
    solver = RobotSolver()

    plan, descr = p_mod_abs(hls, solver, domain, problem, goal=goal, debug=True, n_resamples=3, max_iter=2)

    if type(plan) is str or plan is None:
        sucs.append([goal_info[0][0], 'OPT FAIL'])
        continue
    #import ipdb; ipdb.set_trace()

    if visual:
        agent.add_viewer()


    panda = plan.params['panda']
    for act in plan.actions:
        st, et = act.active_timesteps
        for t in range(st, et):
            grip = panda.right_gripper[:, min(t+1, plan.horizon-1)]
            grip = -0.005 * np.ones(2) if grip[0] < 0.01 else 0.07 * np.ones(2)
            ctrl = np.r_[panda.right[:,t], grip]
            obs, rew, done, info = agent.mjc_env.step(ctrl)
            agent.render_viewer(obs['image'])

    x = agent.get_state()
    goal_suc = [agent.parse_goal(x, g[0], g[1:]) for g in goal_info]
    print('Goal?', goal, goal_suc)
    #import ipdb; ipdb.set_trace()
    sucs.append([goal_info[0], goal_suc])

    print('\n\n-----------\nALL GOALS:')
    for item in sucs:
        print(item)
    print('------------\n\n')

