import copy

import numpy as np

from gps.algorithm.cost.config import COST_STATE
from gps.algorithm.cost.cost import Cost
from gps.algorithm.cost.cost_utils import evall1l2term, get_ramp_multiplier

from policy_hooks.cloth_world_policy_utils import *

class PlaceCost(Cost):
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_STATE)
        config.update(hyperparams)
        Cost.__init__(self, config)
        self.plan = hyperparams['plan']
        self.cloth_tgt = hyperparams['cloth_tgt']

    def update_target(self, cloth):
        self.cloth_tgt = cloth

    def eval(self, sample):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample:  A single sample
        """
        T = sample.T
        Du = sample.dU
        Dx = sample.dX

        final_l = np.zeros(T)
        final_lu = np.zeros((T, Du))
        final_lx = np.zeros((T, Dx))
        final_luu = np.zeros((T, Du, Du))
        final_lxx = np.zeros((T, Dx, Dx))
        final_lux = np.zeros((T, Du, Dx))

        wp = np.zeros((T, 5))
        wp[:, :3] += 1
        wp[:, 3:] += 0.2

        arm_inds = list(range(2,9))
        arm_joints = [self.plan.params['baxter'].openrave_body.GetJointFromDOFIndex(ind) for ind in arm_inds]
        ee_pos = sample.get(EE_ENUM)
        err = np.zeros((T, 5))
        err[:,:3] = ee_pos - sample.get_X()[:, self.plan.state_inds['basket', 'pose']]
        err[4*T/5:,3] = -1*ee_pos + 0.8
        err[:4*T/5,4] = sample.get_X()[:, self.plan.state_inds['baxter', 'lGripper']]
        err[4*T/5:,4] = 0.02 - sample.get_X()[:, self.plan.state_inds['baxter', 'lGripper']]

        jac = np.zeros((T, 5, dX)) # Cloth to gripper distance, gripper distance from target height, gripper joint value
        arm_jac = np.array([np.cross(joint.GetAxis(), ee_pos - joint.GetAnchor(), axisc=0) for joint in arm_joints]).T # TODO: Verify necessary transpose
        base_jac = np.cross(np.array([0, 0, 1]), ee_pos)
        jac[:, :3, self.plan.state_inds['baxter', 'lArmPose']] = arm_jac
        jac[:, :3, self.plan.state_inds['baxter', 'pose']] = base_jac.T
        jac[:, :3, self.plan.state_inds[self.cloth_tgt.name, 'pose']] = -np.eye(e)
        jac[4*T/5:, 3, self.plan.state_inds['baxter', 'lArmPose']] = arm_jac[:, 2]
        jac[:4*T/5, 4, self.plan.state_inds['baxter', 'lGripper']] = -1
        jac[4*T/5:, 4, self.plan.state_inds['baxter', 'lGripper']] = 1

        2nd_jac = np.zeros((T, 5, dX, dX)) # Unused

        l, ls, lss = self._hyperparams['evalnorm'](
            wp, err, jac, jac_2nd, self._hyperparams['l1'],
            self._hyperparams['l2'], self._hyperparams['alpha']
        )

        return l, ls, final_lu, lss, final_luu, final_lux
