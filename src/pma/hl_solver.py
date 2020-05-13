import re
import subprocess
import os
from core.internal_repr.action import Action
from core.internal_repr.plan import Plan

class HLSolver(object):
    """
    HLSolver provides an interface to the chosen task planner.
    """
    def __init__(self, domain_config=None, abs_domain=None):
        self.abs_domain = abs_domain if abs_domain else self._translate_domain(domain_config, first_ts_pre=True)

    def _translate_domain(self, domain_config):
        """
        Translates domain configuration file to representation required for task planner.
        E.g. for an FFSolver this would return a PDDL domain file. Only called once,
        upon creation of an HLSolver object.
        """
        raise NotImplementedError("Override this.")

    def translate_problem(self, concr_prob):
        """
        Translates concrete (instantiated) problem, a Problem object, to representation required for task planner.
        E.g. for an FFSolver this would return a PDDL problem file.
        """
        raise NotImplementedError("Override this.")

    def solve(self, abs_prob, domain, concr_prob, prefix=None):
        """
        Solves the problem and returns a Plan object.

        abs_prob: what got returned by self.translate_problem()
        domain: Domain object
        concr_prob: Problem object
        """
        raise NotImplementedError("Override this.")

class HLState(object):
    """
    Tracks the HL state so that HL state information can be added to preds dict
    attribute in the Action class. For HLSolver use only.
    """
    def __init__(self, init_preds):
        self._pred_dict = {}
        for pred in init_preds:
            rep = HLState.get_rep(pred)
            self._pred_dict[rep] = pred

    def get_preds(self):
        return self._pred_dict.values()

    def in_state(self, pred):
        rep = HLState.get_rep(pred)
        return rep in self._pred_dict

    def update(self, pred_dict_list):
        for pred_dict in pred_dict_list:
            self.add_pred_from_dict(pred_dict)

    def add_pred_from_dict(self, pred_dict):
        if pred_dict["hl_info"] is "eff":
            negated = pred_dict["negated"]
            pred = pred_dict["pred"]
            rep = HLState.get_rep(pred)
            if negated and self.in_state(pred):
                del self._pred_dict[rep]
            elif not negated and not self.in_state(pred):
                self._pred_dict[rep] = pred

    @staticmethod
    def get_rep(pred):
        s = "(%s "%(pred.get_type())
        for param in pred.params[:-1]:
            s += param.name + " "
        s += pred.params[-1].name + ")"
        return s

class FFSolver(HLSolver):
    FF_EXEC = "../task_planners/FF-v2.3/ff"
    FILE_PREFIX = "temp_"

    def _parse_precondition_ts(self, pre, ts):
        preds = ''
        so_far = []
        ts = ts.strip().split()
        ts = [t.split(':') for t in ts]
        ts = [(int(t[0]), int(t[1])) for t in ts]

        count, inds = 0, [0]
        pre = pre[5:-1]
        for i, token in enumerate(pre):
            if token == "(":
                count += 1
            if token == ")":
                count -= 1
                if count == 0:
                    inds.append(i+1)
        for i in range(len(inds)):
            if ts[i][0] == 0 and ts[i][1] == 0:
                pred = pre[inds[i]:inds[i+1]] if i+1 < len(inds) else pre[inds[i]:]
                if pred not in so_far:
                    preds += pred
                    so_far.append(pred)

        return '(and {0})'.format(preds)

    def _parse_exclude(self, preds):
        parsed = []
        preds = preds[4:-1].strip()
        count, ind = 0, 0
        for i, token in enumerate(preds):
            if token == "(":
                count += 1
            if token ==")":
                count -= 1
                if count == 0:
                    parsed.append(preds[ind:i+1].strip())
                    ind = i + 1

        for i, pred in enumerate(parsed):
            if pred.find('/') >= 0:
                new_pred = ''
                cur_pred = pred
                eq = ''
                depth = 0
                while cur_pred.find('forall') >= 0:
                    depth += 1
                    new_pred += '(forall '
                    m = re.match("\(\s*forall", cur_pred)
                    cur_pred = cur_pred[m.span()[1]:-1].strip()
                    g = re.match("\((.*?)\)(.*)", cur_pred).groups()
                    v = g[0].split("/")
                    quant = '({0}) '.format(v[0].strip())
                    for e in v[1:]:
                        eq += '(not (= {0} {1}))'.format(v[0].split('-')[0].strip(), e.strip())
                    cond = g[1].strip()
                    new_pred += quant
                    if cond.find('forall') >= 0:
                        cur_pred = cond
                    else:
                        new_pred += '(when {0} {1})'.format(eq, cond)
                eq += ''
                for _ in range(depth):
                    new_pred += ')'
                parsed[i] = new_pred
        out = '(and '
        for step in parsed:
            out += step + ' '
        out += ')'
        return out

    def _translate_domain(self, domain_config, first_ts_pre=False):
        """
        Argument:
            domain_config: parsed domain configuration that defines the problem
                        (Dict\{String: String\})
        Return:
            translated domain in .PDDL recognizable by HLSolver (String)
        """
        dom_str = "; AUTOGENERATED. DO NOT EDIT.\n\n(define (domain robotics)\
                  \n(:requirements :strips :equality :typing)\n(:types "
        for t in domain_config["Types"].split(";"):
            dom_str += t.strip().split("(")[0].strip() + " "
        dom_str += ")\n\n(:predicates\n"
        for p_defn in domain_config["Derived Predicates"].split(";"):
            p_name, p_params = map(str.strip, p_defn.split(",", 1))
            p_params = [s.strip() for s in p_params.split(",")]
            dom_str += "(%s "%p_name
            for i, param in enumerate(p_params):
                dom_str += "?var%d - %s "%(i, param)
            dom_str += ")\n"
        dom_str += ")\n\n"
        for key in domain_config.keys():
            if key.startswith("Action"):
                count, inds = 0, [0]
                for i, token in enumerate(domain_config[key]):
                    if token == "(":
                        count += 1
                    if token == ")":
                        count -= 1
                        if count == 0:
                            inds.append(i+1)
                params = domain_config[key][inds[0]:inds[1]].strip()
                pre = domain_config[key][inds[1]:inds[2]].strip()
                eff = domain_config[key][inds[2]:inds[3]].strip()
                pre = self._parse_exclude(pre)
                eff = self._parse_exclude(eff)
                if first_ts_pre:
                    ts = domain_config[key][inds[3]:].strip()
                    pre = self._parse_precondition_ts(pre, ts)
                dom_str += "(:action %s\n:parameters %s\n:precondition %s\n:effect %s\n)\n\n"%(key.split()[1], params, pre, eff)
        dom_str += ")"

        # This block is added to erase domain name
        clean_str = ""
        if "Baxter" in dom_str:
            for word in dom_str.split("Baxter"):
                clean_str += word
            dom_str = clean_str
        elif "PR2" in dom_str:
            for word in dom_str.split("PR2"):
                clean_str += word
            dom_str = clean_str
        return dom_str

    def translate_problem(self, concr_prob, initial=None, goal=None):
        """
        Argument:
            concr_prob: problem that defines initial state and goal configuration.
                        (internal_repr/problem)
        Return:
            translated problem in .PDDL recognizable by HLSolver (String)
        """
        prob_str = "; AUTOGENERATED. DO NOT EDIT.\n\n(define (problem ff_prob)\n(:domain robotics)\n(:objects\n"
        for param in concr_prob.init_state.params.values():
            prob_str += "%s - %s\n"%(param.name, param.get_type())
        prob_str += ")\n\n(:init\n"
        if initial is None:
            for pred in concr_prob.init_state.preds:
                prob_str += "(%s "%pred.get_type()
                for param in pred.params:
                    prob_str += "%s "%param.name
                prob_str += ")\n"
        else:
            initial = set(initial)
            for pred in initial:
                prob_str += pred
            concr_prob.initial = initial
        prob_str += ")\n\n(:goal\n(and "
        if goal is None:
            for pred in concr_prob.goal_preds:
                prob_str += "(%s "%pred.get_type()
                for param in pred.params:
                    prob_str += "%s "%param.name
                prob_str += ") "
        else:
            for pred in goal:
                prob_str += pred
            concr_prob.goal = goal
        prob_str += ")\n)\n)"
        # This block is added to erase domain name
        clean_str = ""
        if "Baxter" in prob_str:
            for word in prob_str.split("Baxter"):
                clean_str += word
            prob_str = clean_str
        elif "PR2" in prob_str:
            for word in prob_str.split("PR2"):
                clean_str += word
            prob_str = clean_str
        return prob_str

    def solve(self, abs_prob, domain, concr_prob, prefix=None, label=''):
        """
        Argument:
            abs_prob: translated problem in .PDDL recognizable by HLSolver (String)
            domain: domain in which problem is defined. (internal_repr/domain)
            concr_prob: problem that defines initial state and goal configuration.
                        (internal_repr/problem)
        Return:
            Plan Object for ll_solver to optimize. (internal_repr/plan)
        """
        plan_str = self._run_planner(self.abs_domain, abs_prob, label=label)
        if plan_str == Plan.IMPOSSIBLE:
            return plan_str

        if prefix:
            for i in range(len(plan_str)):
                step, action = plan_str[i].split(':')
                plan_str[i] = str(len(prefix) + int(step)) + ':' + action
            plan_str = prefix + plan_str
        plan = self.get_plan(plan_str, domain, concr_prob)
        if type(plan) is not str:
            plan.plan_str = plan_str
            plan.goal = concr_prob.goal
            plan.initial = concr_prob.initial
        return plan

    def get_plan(self, plan_str, domain, concr_prob):
        """
        Argument:
            plan_str: list of high level plan. (List(String))
            domain: domain in which problem is defined. (internal_repr/domain)
            concr_prob: problem that defines initial state and goal configuration.
                        (internal_repr/problem)
        Return:
            Plan Object for ll_solver to optimize. (internal_repr/plan)

        Note: Actions, and Parameters are created here.
        """
        if plan_str == Plan.IMPOSSIBLE:
            return plan_str
        openrave_env = concr_prob.env
        plan_horizon = self._extract_horizon(plan_str, domain)
        params = self._spawn_plan_params(concr_prob, plan_horizon)
        actions = self._spawn_actions(plan_str, domain, params,
                                      plan_horizon, concr_prob, openrave_env)
        plan = Plan(params, actions, plan_horizon, openrave_env)
        plan.start_action = concr_prob.start_action
        plan.prob = concr_prob
        plan.domain = domain
        return plan


    def _extract_horizon(self, plan_str, domain):
        """
        Argument:
            plan_str: list of high level plan. (List(String))
            domain: domain in which problem is defined. (internal_repr/domain)
        Return:
            planning horizon for the entire plan. (Integer)
        """
        hor = 1
        for action_str in plan_str:
            spl = action_str.split()
            a_name = spl[1].lower()
            ## subtract 1 b/c subsequent actions have an overlapping
            ## first and last state
            hor += domain.action_schemas[a_name].horizon - 1
        return hor

    def _spawn_plan_params(self, concr_prob, plan_horizon):
        """
        Argument:
            concr_prob: problem that defines initial state and goal configuration.
                        (internal_repr/problem)
            plan_horizon: planning horizon for the entire plan. (Integer)
        Return:
            A mapping between parameter name and parameter
            (Dict\{String: internal_repr/parameter\})
        """
        params = {}
        for p_name, p in concr_prob.init_state.params.items():
            params[p_name] = p.copy(plan_horizon, True)
        return params

    def _spawn_actions(self, plan_str, domain, params,
                                       plan_horizon, concr_prob, env):
        """
        Argument:
            plan_str: list of high level plan. (List(String))
            domain: domain in which problem is defined. (internal_repr/domain)
            params: dictionary mapping name to parameter.
                    (Dict\{String: internal_repr/parameter\})
            plan_horizon: planning horizon for the entire plan. (Integer)
            concr_prob: problem that defines initial state and goal configuration.
                        (internal_repr/problem)
            env: Openrave Environment for planning (openravepy/Environment)
        Return:
            list of actions to plan over (List(internal_repr/action))
        """
        actions = []
        curr_h = 0
        hl_state = HLState(concr_prob.init_state.preds)
        for action_str in plan_str:
            spl = action_str.split()
            step = int(spl[0].split(":")[0])
            a_name, a_args = spl[1].lower(), map(str.lower, spl[2:])
            a_schema = domain.action_schemas[a_name]
            var_names, expected_types = zip(*a_schema.params)
            bindings = dict(zip(var_names, zip(a_args, expected_types)))
            preds = []
            for p_d in a_schema.preds:
                pred_schema = domain.pred_schemas[p_d["type"]]
                arg_valuations = [[]]
                for a in p_d["args"]:
                    if a in bindings:
                        # if we have a binding, add the arg name to all valuations
                        for val in arg_valuations:
                            val.append(bindings[a])
                    else:
                        # handle universally quantified params by creating a valuation for each possibility
                        excl = [bindings[e][0] for e in bindings if e in a_schema.exclude_params[a]]
                        p_type = a_schema.universally_quantified_params[a]
                        bound_names = [bindings[key][0] for key in bindings]
                        # arg_names_of_type = [k for k, v in params.items() if v.get_type() == p_type and k not in bound_names]
                        arg_names_of_type = [k for k, v in params.items() if v.get_type() == p_type and k not in excl]
                        arg_valuations = [val + [(name, p_type)] for name in arg_names_of_type for val in arg_valuations]
                for val in arg_valuations:
                    val, types = zip(*val)
                    try:
                        assert list(types) == pred_schema.expected_params, "Expected params from schema don't match types! Bad task planner output."
                    except:
                        import ipdb; ipdb.set_trace()
                    # if list(types) != pred_schema.expected_params:
                    #     import pdb; pdb.set_trace()
                    pred = pred_schema.pred_class("placeholder", [params[v] for v in val], pred_schema.expected_params, env=env)
                    ts = (p_d["active_timesteps"][0] + curr_h, p_d["active_timesteps"][1] + curr_h)
                    preds.append({"negated": p_d["negated"], "hl_info": p_d["hl_info"], "active_timesteps": ts, "pred": pred})
            # adding predicates from the hl state to action's preds
            action_pred_rep = [HLState.get_rep(pred_dict["pred"]) for pred_dict in preds]
            for pred in hl_state.get_preds():
                if HLState.get_rep(pred) not in action_pred_rep:
                    preds.append({"negated": False, "hl_info": "hl_state", "active_timesteps": (curr_h, curr_h + a_schema.horizon - 1), "pred": pred})
            # updating hl_state
            hl_state.update(preds)
            actions.append(Action(step, a_name, (curr_h, curr_h + a_schema.horizon - 1), [params[arg] for arg in a_args], preds))
            curr_h += a_schema.horizon - 1
        return actions



    def _run_planner(self, abs_domain, abs_prob, label=''):
        """
        Argument:
            abs_domain: translated domain in .PDDL recognizable by HLSolver (String)
            abs_prob: translated problem in .PDDL recognizable by HLSolver (String)
        Return:
            list of high level plan (List(String))
        Note:
            High level planner gets called here.
        """
        if not os.path.isdir('temp'):
            os.mkdir('temp')
        fprefix = 'temp/'+label+'_'+FFSolver.FILE_PREFIX
        with open("%sdom.pddl"%(fprefix), "w") as f:
            f.write(abs_domain)
        with open("%sprob.pddl"%(fprefix), "w") as f:
            f.write(abs_prob)
        with open("%sprob.output"%(fprefix), "w") as f:
            subprocess.call([FFSolver.FF_EXEC, "-o", "%sdom.pddl"%(fprefix), "-f", "%sprob.pddl"%(fprefix)], stdout=f)
        with open("%sprob.output"%(fprefix), "r") as f:
            s = f.read()
        if "goal can be simplified to FALSE" in s or "problem proven unsolvable" in s:
            # import ipdb; ipdb.set_trace()
            plan = Plan.IMPOSSIBLE
        else:
            try:
                plan = filter(lambda x: x, map(str.strip, s.split("found legal plan as follows")[1].split("time")[0].replace("step", "").split("\n")))
            except:
                print('Error in filter for', s, fprefix)
                plan = Plan.IMPOSSIBLE

        '''
        subprocess.call(["rm", "-f", "%sdom.pddl"%fprefix,
                         "%sprob.pddl"%fprefix,
                         "%sprob.pddl.soln"%fprefix,
                         "%sprob.output"%fprefix])
        '''
        if plan != Plan.IMPOSSIBLE:
            plan = self._patch_redundancy(plan)
        return plan

    def _patch_redundancy(self, plan_str):
        """
        Argument:
            plan_str: list of high level plan (List(String))
        Return:
            list of high level plan that don't have redundancy. (List(String))
        """
        i = 0
        while i < len(plan_str)-1:
            if "MOVETO" in plan_str[i] and "MOVETO" in plan_str[i+1]:
                pose = plan_str[i+1].split()[-1]
                del plan_str[i+1]
                spl = plan_str[i].split()
                spl[-1] = pose
                plan_str[i] = " ".join(spl)
            else:
                i += 1
        for i in range(len(plan_str)):
            spl = plan_str[i].split(":", 1)
            plan_str[i] = "%s:%s"%(i, spl[1])
        return plan_str

class DummyHLSolver(HLSolver):
    def _translate_domain(self, domain_config):
        return "translate domain"

    def translate_problem(self, concr_prob):
        return "translate problem"

    def solve(self, abs_prob, domain, concr_prob, prefix=None):
        return "solve"
