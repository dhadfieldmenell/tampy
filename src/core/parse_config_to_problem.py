from IPython import embed as shell
import init_env
from internal_repr import parameter
from internal_repr import predicate
from internal_repr import state
from internal_repr import problem

class ParseConfigToProblem:
    """
    Read the configuration data and spawn the corresponding initial Problem object (see Problem class).
    """
    def __init__(self, config):
        self.config = config

    def parse(self):
        s = self.config["Environment Initializer"]
        if not hasattr(init_env, s):
            raise Exception("Environment Initializer '%s' not defined!"%s)
        initer = getattr(init_env, s)()

        # create parameter objects
        params = {}
        for param in self.config["Objects"].split(","):
            p_name, p_type = map(str.strip, param.split("-"))
            if not hasattr(parameter, p_type):
                raise Exception("Parameter type '%s' not defined!"%p_type)
            params[p_name] = getattr(parameter, p_type)("hl_" + p_name)

        # create initial state predicate objects
        preds_to_param_types = {}
        for p_defn in self.config["Predicates"].split(";"):
            p_name, p_type = map(str.strip, p_defn.split(",", 1))
            preds_to_param_types[p_name] = [s.strip() for s in p_type.split(",")]
        init_preds = set()
        for i, pred in enumerate(self.config["Init"].split(",")):
            spl = map(str.strip, pred.strip("() ").split())
            p_name, p_args = spl[0], spl[1:]
            if not hasattr(predicate, p_name):
                raise Exception("Predicate type '%s' not defined!"%p_name)
            init_preds.add(getattr(predicate, p_name)(name="hlinitpred%d"%i,
                                                      params=[params[n] for n in p_args],
                                                      expected_param_types=preds_to_param_types[p_name]))

        # call env initializer to populate parameter tables (with only 1 timestep of data)
        env_data = initer.construct_env_and_init_params(self.config["Environment File"], params, init_preds)

        # use preds and their params to create an initial State object
        initial_state = state.State("initstate", init_preds, timestep=0)

        # create goal predicate objects
        goal_preds = set()
        for i, pred in enumerate(self.config["Goal"].split(",")):
            spl = map(str.strip, pred.strip("() ").split())
            p_name, p_args = spl[0], spl[1:]
            if not hasattr(predicate, p_name):
                raise Exception("Predicate type '%s' not defined!"%p_name)
            goal_preds.add(getattr(predicate, p_name)(name="hlgoalpred%d"%i,
                                                      params=[params[n] for n in p_args],
                                                      expected_param_types=preds_to_param_types[p_name]))

        # use initial state to create Problem object
        initial_problem = problem.Problem(initial_state, goal_preds, env_data, time_horizon=0)
        return initial_problem
