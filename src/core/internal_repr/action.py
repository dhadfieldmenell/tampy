from IPython import embed as shell

class Action(object):
    """
    An instantiated action stores the following.

    name: name of this action
    horizon: number of allotted timesteps
    TODO: figure this out
    """
    def __init__(self, name, num_timesteps, params, preds, pred_times):
        self.name = name
        self.num_timesteps = num_timesteps
        self.params = params
        self.preds = preds
        self.pred_times = pred_times
