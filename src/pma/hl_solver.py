from IPython import embed as shell
import subprocess
from core.internal_repr.action import Action
from core.internal_repr.plan import Plan

class HLSolver(object):
    """
    HLSolver provides an interface to the chosen task planner.
    """
    def __init__(self, domain_config):
        self.abs_domain = self._translate_domain(domain_config)

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

    def solve(self, abs_prob, domain, concr_prob):
        """
        Solves the problem and returns a Plan object.

        abs_prob: what got returned by self.translate_problem()
        domain: Domain object
        concr_prob: Problem object
        """
        raise NotImplementedError("Override this.")

class FFSolver(HLSolver):
    FF_EXEC = "../task_planners/FF-v2.3/ff"
    FILE_PREFIX = "temp_"

    def _translate_domain(self, domain_config):
        dom_str = "; AUTOGENERATED. DO NOT EDIT.\n\n(define (domain robotics)\n(:requirements :strips :equality :typing)\n(:types "
        for t in domain_config["Types"].split(";"):
            dom_str += t.strip().split("(")[0].strip() + " "
        dom_str += ")\n\n(:predicates\n"
        for p_defn in domain_config["Predicates"].split(";"):
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
                assert len(inds) == 4
                params = domain_config[key][inds[0]:inds[1]].strip()
                pre = domain_config[key][inds[1]:inds[2]].strip()
                eff = domain_config[key][inds[2]:inds[3]].strip()
                dom_str += "(:action %s\n:parameters %s\n:precondition %s\n:effect %s\n)\n\n"%(key.split()[1], params, pre, eff)
        dom_str += ")"
        return dom_str

    def translate_problem(self, concr_prob):
        prob_str = "; AUTOGENERATED. DO NOT EDIT.\n\n(define (problem ff_prob) (:domain robotics)\n(:objects\n"
        for param in concr_prob.init_state.params.values():
            prob_str += "%s - %s\n"%(param.name, param.get_type())
        prob_str += ")\n\n(:init\n"
        for pred in concr_prob.init_state.preds:
            prob_str += "(%s "%pred.get_type()
            for param in pred.params:
                prob_str += "%s "%param.name
            prob_str += ")\n"
        prob_str += ")\n\n(:goal\n(and "
        for pred in concr_prob.goal_preds:
            prob_str += "(%s "%pred.get_type()
            for param in pred.params:
                prob_str += "%s "%param.name
            prob_str += ") "
        prob_str += ")\n)\n)"
        return prob_str

    def solve(self, abs_prob, domain, concr_prob):
        plan_str = self._run_planner(self.abs_domain, abs_prob)
        if plan_str == Plan.IMPOSSIBLE:
            return plan_str
        plan_horizon = self._extract_horizon(plan_str, domain)
        params = self._spawn_plan_params(concr_prob, plan_horizon)
        actions = self._spawn_actions(plan_str, domain, params, plan_horizon)
        return Plan(params, actions, plan_horizon)

    def _extract_horizon(self, plan_str, domain):
        hor = 0
        for action_str in plan_str:
            spl = action_str.split()
            a_name = spl[1].lower()
            hor += domain.action_schemas[a_name].horizon
        return hor

    def _spawn_plan_params(self, concr_prob, plan_horizon):
        params = {}
        for p_name, p in concr_prob.init_state.params.items():
            params[p_name] = p.copy(plan_horizon)
        return params

    def _spawn_actions(self, plan_str, domain, params, plan_horizon):
        actions = []
        curr_h = 0
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
                        p_type = a_schema.universally_quantified_params[a]
                        arg_names_of_type = [k for k, v in params.items() if v.get_type() == p_type]
                        arg_valuations = [val + [(name, p_type)] for name in arg_names_of_type for val in arg_valuations]
                for val in arg_valuations:
                    val, types = zip(*val)
                    assert list(types) == pred_schema.expected_params, "Expected params from schema don't match types! Bad task planner output."
                    pred = pred_schema.pred_class("placeholder", [params[v] for v in val], pred_schema.expected_params)
                    ts = (p_d["active_timesteps"][0] + curr_h, p_d["active_timesteps"][1] + curr_h)
                    preds.append({"negated": p_d["negated"], "active_timesteps": ts, "pred": pred})
            actions.append(Action(step, a_name, (curr_h, curr_h + a_schema.horizon - 1), [params[arg] for arg in a_args], preds))
            curr_h += a_schema.horizon
        return actions

    def _run_planner(self, abs_domain, abs_prob):
        with open("%sdom.pddl"%FFSolver.FILE_PREFIX, "w") as f:
            f.write(abs_domain)
        with open("%sprob.pddl"%FFSolver.FILE_PREFIX, "w") as f:
            f.write(abs_prob)
        with open("%sprob.output"%FFSolver.FILE_PREFIX, "w") as f:
            subprocess.call([FFSolver.FF_EXEC, "-o", "%sdom.pddl"%FFSolver.FILE_PREFIX, "-f", "%sprob.pddl"%FFSolver.FILE_PREFIX], stdout=f)
        with open("%sprob.output"%FFSolver.FILE_PREFIX, "r") as f:
            s = f.read()
        if "goal can be simplified to FALSE" in s or "problem proven unsolvable" in s:
            plan = Plan.IMPOSSIBLE
        else:
            plan = filter(lambda x: x, map(str.strip, s.split("found legal plan as follows")[1].split("time")[0].replace("step", "").split("\n")))
        subprocess.call(["rm", "-f", "%sdom.pddl"%FFSolver.FILE_PREFIX,
                         "%sprob.pddl"%FFSolver.FILE_PREFIX,
                         "%sprob.pddl.soln"%FFSolver.FILE_PREFIX,
                         "%sprob.output"%FFSolver.FILE_PREFIX])
        if plan != Plan.IMPOSSIBLE:
            plan = self._patch_redundancy(plan)
        return plan

    def _patch_redundancy(self, plan_str):
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

    def solve(self, abs_prob, domain, concr_prob):
        return "solve"
