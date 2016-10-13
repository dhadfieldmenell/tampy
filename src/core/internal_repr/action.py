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

    def get_failed_preds(self, active_ts=None):
        if active_ts is None:
            active_ts = self.active_timesteps
        failed = []
        for pred_d in self.preds:
            if pred_d['hl_info'] == 'hl_state': continue
            pred = pred_d['pred']
            negated = pred_d['negated']
            start, end = pred_d['active_timesteps']
            for t in range(max(start, active_ts[0]),
                           min(end, active_ts[1])+1):
                if not pred.test(t, negated=negated):
                    failed.append((negated, pred, t))
        return failed

    def get_active_preds(self, t):
        res = []
        for pred_d in self.preds:
            if pred_d['hl_info'] == 'hl_state': continue
            start, end = pred_d['active_timesteps']
            if start <= t and end >= t: res.append(pred_d['pred'])
        return res

    def satisfied(self, active_ts=None):
        if active_ts is None:
            active_ts = self.active_timesteps
        elif self.active_timesteps[0] >= active_ts[1] \
            or self.active_timesteps[1] <= active_ts[0]:
            return True
        return len(self.get_failed_preds(active_ts)) == 0

    def first_failed_ts(self):
        start, end = self.active_timesteps
        ## init at the maximize
        t_min = end
        for b, p, t in self.get_failed_preds():
            if t < t_min:
                t_min = t
        return t_min

    def copy(self, start_ts, param_copy_dict, preds=None):
        start, end = self.active_timesteps
        active_timesteps = start_ts, end - start + start_ts
        params = [param_copy_dict[param] for param in self.params]
        if preds is None:
            preds = self.preds
        act_preds = []
        for pred_dict in preds:
            pred_start, pred_end = pred_dict['active_timesteps']
            if pred_start > end or pred_end < start:
                continue
            pred_start = max(start, pred_start)
            pred_end = min(end, pred_end)

            pred_dict_copy = {}
            pred_dict_copy['active_timesteps'] = (pred_start - start + start_ts,
                                                  pred_end - start + start_ts)
            pred_dict_copy['pred'] = pred_dict['pred'].copy(param_copy_dict)
            pred_dict_copy['negated'] = pred_dict['negated']
            pred_dict_copy['hl_info'] = pred_dict['hl_info']
            act_preds.append(pred_dict_copy)
        return Action(self.step_num, self.name, active_timesteps, params, act_preds)
