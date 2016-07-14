from IPython import embed as shell
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

    def __init__(self, params, actions, horizon, env):
        self.params = params
        self.actions = actions
        self.horizon = horizon
        self.env = env
        self.initialized = False
        self._free_params = {}
        self._determine_free_attrs()

    def _determine_free_attrs(self):
        for p in self.params.itervalues():
            for k, v in p.__dict__.iteritems():
                if type(v) == np.ndarray:
                    ## free variables are indicated as numpy arrays of NaNs
                    arr = np.zeros(v.shape, dtype=np.int)
                    arr[np.isnan(v)] = 1
                    p._free_attrs[k] = arr

    def execute(self):
        raise NotImplementedError

    def get_failed_pred(self):
        raise NotImplementedError

    def satisfied(self):
        raise NotImplementedError
