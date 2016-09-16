from IPython import embed as shell
from action import Action
import numpy as np

class Plan(object):
    """
    A plan has the following.

    params: dictionary of plan parameters, mapping name to object
    actions: list of Actions
    horizon: total number of timesteps for plan

    This class also defines methods for executing actions in simulation using the chosen viewer.
    """
    IMPOSSIBLE = "Impossible"

    def __init__(self, params, actions, horizon, env, determine_free=True):
        self.params = params
        self.actions = actions
        self.horizon = horizon
        self.env = env
        self.initialized = False
        self._free_attrs = {}
        self._saved_free_attrs = {}
        if determine_free:
            self._determine_free_attrs()

    @staticmethod
    def create_plan_for_preds(preds, env):
        ## preds is a list of pred, negated
        ## constructs a plan with a single action that
        ## enforces all the preds
        p_dicts = []
        params = set()
        for p, neg in preds:
            p_dicts.append({'pred': p, 'hl_info': 'pre', 'active_timesteps': (0, 0),
                            'negated': neg})
            params = params.union(p.params)
        params = list(params)
        a = Action(0, 'dummy', (0,0), params, p_dicts)
        param_dict = dict([(p.name, p) for p in params])
        return Plan(param_dict, [a], 1, env, determine_free=False)

    def _determine_free_attrs(self):
        for p in self.params.itervalues():
            for k, v in p.__dict__.iteritems():
                if type(v) == np.ndarray:
                    ## free variables are indicated as numpy arrays of NaNs
                    arr = np.zeros(v.shape, dtype=np.int)
                    arr[np.isnan(v)] = 1
                    p._free_attrs[k] = arr

    def save_free_attrs(self):
        for p in self.params.itervalues():
            p.save_free_attrs()

    def restore_free_attrs(self):
        for p in self.params.itervalues():
            p.restore_free_attrs()

    def execute(self):
        raise NotImplementedError

    def get_param(self, pred_type, target_ind, partial_assignment = None,
                  negated=False, return_preds=False):
        """
        get all target_ind parameters of the given predicate type
        partial_assignment is a dict that maps indices to parameter
        """
        if partial_assignment is None:
            partial_assignment = {}
        res = []
        if return_preds:
            preds = []
        for p in self.get_preds(incl_negated = negated):
            has_partial_assignment = True
            if p.get_type() != pred_type: continue
            for idx, v in partial_assignment.iteritems():
                if p.params[idx] != v:
                    has_partial_assignment = False
                    break
            if has_partial_assignment:
                res.append(p.params[target_ind])
                if return_preds: preds.append(p)
        res = np.unique(res)
        if return_preds:
            return res, np.unique(preds)
        return res

    def get_preds(self, incl_negated):
        res = []
        for a in self.actions:
            if incl_negated:
                res.extend([p['pred'] for p in a.preds if p['hl_info'] != 'hl_state'])
            else:
                res.extend([p['pred'] for p in a.preds if p['hl_info'] != 'hl_state' and not p['negated']])

        return res

    def get_failed_pred(self, active_ts=None):
        #just return the first one for now
        t_min = self.horizon+1
        pred = None
        negated = False
        for n, p, t in self.get_failed_preds(active_ts=active_ts):
            if t < t_min:
                t_min = t
                pred = p
                negated = n
        return negated, pred, t_min

    def get_failed_preds(self, active_ts=None):
        if active_ts == None:
            active_ts = (0, self.horizon-1)
        failed = []
        for a in self.actions:
            failed.extend(a.get_failed_preds(active_ts))
        return failed

    def satisfied(self, active_ts=None):
        if active_ts == None:
            active_ts = (0, self.horizon-1)
        success = True
        for a in self.actions:
            success &= a.satisfied(active_ts)
        return success

    def get_active_preds(self, t):
        res = []
        for a in self.actions:
            start, end = a.active_timesteps
            if start <= t and end >= t:
                res.extend(a.get_active_preds(t))
        return res

    def get_action_plans(self):
        plans = []
        param_to_copies = {}
        param_to_name = {param: name for name, param in self.params.items()}
        for a in self.actions:
            active_timesteps = a.active_timesteps
            start, end = active_timesteps
            action_param_to_copy = {}
            for param in a.params:
                param_copy = param.copy_ts(active_timesteps)
                action_param_to_copy[param] = param_copy

                param_ts_tuple = (start, param_copy)
                if param in param_to_copies:
                    param_to_copies[param].append(param_ts_tuple)
                else:
                    param_to_copies[param] = [param_ts_tuple]

            actions = [a.copy(0, action_param_to_copy)]
            horizon = end - start + 1

            params_dict = {param_to_name[param]: param_copy \
                            for param, param_copy in action_param_to_copy.items()}
            plans.append(Plan(params_dict, actions, horizon, self.env))
        return plans, param_to_copies
