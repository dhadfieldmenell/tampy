from IPython import embed as shell

class Action(object):
    """
    An instantiated action stores the following.

    step_num: step number of this action in the plan
    name: name of this action
    active_timesteps: (start_time, end_time) for this action
    params: ordered list of Parameter objects
    preds: list of dictionaries where each dictionary contains information about
    a predicate. each dictionary contains
    - the Predicate object
    - negated (Boolean)
    - hl_info (string) which is "pre", "post" and "hl_state" if the predicate is
      a precondition, postcondition, or neither and part of the high level state
      respectively
    - active_timesteps (tuple of (start_time, end_time))
    """
    def __init__(self, step_num, name, active_timesteps, params, preds):
        self.step_num = step_num
        self.name = name
        self.active_timesteps = active_timesteps
        self.params = params
        self.preds = preds

    def __repr__(self):
        return "%d: %s %s %s"%(self.step_num, self.name, self.active_timesteps, " ".join([p.name for p in self.params]))
