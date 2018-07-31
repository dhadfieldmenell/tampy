def ts_constr_violations(plan, ts, include=[]):
    preds = [(pred, negated) for negated, pred, t in plan.get_failed_preds(active_ts=(ts,ts), priority=3, tol=1e-3)]
    if len(include): preds = filter(lambda p: p[0].name in include, preds)
    return {pred.name: np.max(pred.check_pred_violation(t, negated=negated, tol=1e-3)) for pred, negated in preds}

def violated_ll_constrs(plan, ts, include=[]):
    return [pred.name for negated, pred, t in plan.get_failed_preds(active_ts=(ts,ts), priority=3, tol=1e-3)]

def violated_hl_constrs(plan, ts, hl_eval_funcs):
    return [f(plan, ts) for f in hl_eval_funcs]

