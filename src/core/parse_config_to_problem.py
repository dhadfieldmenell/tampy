from IPython import embed as shell
from core.internal_repr import state
from core.internal_repr import problem

class ParseConfigToProblem(object):
    """
    Read the problem configuration data and spawn the corresponding initial Problem object (see Problem class).
    This is only done for spawning the very first Problem object, from the initial state in the problem configuration file.
    The domain is passed in for validation against the schema.
    """
    def __init__(self, problem_config, domain):
        self.problem_config = problem_config
        self.domain = domain

    def parse(self):
        # create parameter objects
        params = {}
        for t in self.problem_config["Objects"].split(";"):
            o_type, attrs = map(str.strip, t.strip(" )").split("(", 1))
            attr_dict = dict([l.split(" ", 1) for l in map(str.strip, attrs.split("."))])
            attr_dict["_type"] = o_type
            try:
                params[attr_dict["name"]] = self.domain.param_schema[o_type][0](attrs=attr_dict,
                                                                                attr_types=self.domain.param_schema[o_type][1])
            except KeyError:
                raise Exception("Parameter '%s' not defined in domain file."%o_type)

        # create initial state predicate objects
        init_preds = set()
        for i, pred in enumerate(self.problem_config["Init"].split(",")):
            spl = map(str.strip, pred.strip("() ").split())
            p_name, p_args = spl[0], spl[1:]
            init_preds.add(self.domain.pred_schema[p_name][0](name="initpred%d"%i,
                                                              params=[params[n] for n in p_args],
                                                              expected_param_types=self.domain.pred_schema[p_name][1]))

        # use params and initial preds to create an initial State object
        initial_state = state.State("initstate", params.values(), init_preds, timestep=0)

        # create goal predicate objects
        goal_preds = set()
        for i, pred in enumerate(self.problem_config["Goal"].split(",")):
            spl = map(str.strip, pred.strip("() ").split())
            p_name, p_args = spl[0], spl[1:]
            goal_preds.add(self.domain.pred_schema[p_name][0](name="goalpred%d"%i,
                                                              params=[params[n] for n in p_args],
                                                              expected_param_types=self.domain.pred_schema[p_name][1]))

        # use initial state to create Problem object
        initial_problem = problem.Problem(initial_state, goal_preds)
        return initial_problem
