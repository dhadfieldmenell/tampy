from IPython import embed as shell
from core.internal_repr import state
from core.internal_repr import problem
from errors_exceptions import ProblemConfigException

class ParseProblemConfig(object):
    """
    Read the problem configuration data and spawn the corresponding initial Problem object (see Problem class).
    This is only done for spawning the very first Problem object, from the initial state specified in the problem configuration file.
    Validation is performed against the schemas stored in the Domain object self.domain.
    """
    @staticmethod
    def parse(problem_config, domain):
        # create parameter objects
        params = {}
        if "Objects" not in problem_config or not problem_config["Objects"]:
            raise ProblemConfigException("Problem file needs objects.")
        for t in problem_config["Objects"].split(";"):
            o_type, attrs = map(str.strip, t.strip(" )").split("(", 1))
            attr_dict = dict([l.split(" ", 1) for l in map(str.strip, attrs.split("."))])
            assert "name" in attr_dict
            attr_dict["_type"] = o_type
            params[attr_dict["name"]] = attr_dict

        if "Init" not in problem_config or not problem_config["Init"]:
            raise ProblemConfigException("Problem file needs init.")
        prim_preds, deriv_preds = map(str.strip, problem_config["Init"].split(";"))
        if prim_preds:
            for pred in map(str.strip, prim_preds.split(")")):
                if pred:
                    a, b = pred.find("["), pred.rfind("]") + 1
                    if a != -1:
                        new_s = "".join(pred[a:b].split())
                        pred = pred.replace(pred[a:b], new_s)
                    k, obj_name, v = map(str.strip, pred.strip(",() ").split())
                    if obj_name not in params:
                        raise ProblemConfigException("'%s' is not an object in problem file."%obj_name)
                    params[obj_name][k] = v.replace("[", "(").replace("]", ")")
            for obj_name, attr_dict in params.items():
                assert "pose" in attr_dict or "value" in attr_dict
                o_type = attr_dict["_type"]
                try:
                    params[obj_name] = domain.param_schemas[o_type].param_class(attrs=attr_dict,
                                                                                attr_types=domain.param_schemas[o_type].attr_dict)
                except KeyError:
                    raise ProblemConfigException("Parameter '%s' not defined in domain file."%attr_dict["name"])
                except ValueError:
                    raise ProblemConfigException("Some attribute type in parameter '%s' is incorrect."%attr_dict["name"])
        for k, v in params.items():
            if type(v) is dict:
                raise ProblemConfigException("Problem file has no primitive predicates for object '%s'."%k)
        init_preds = set()
        if deriv_preds:
            for i, pred in enumerate(deriv_preds.split(",")):
                spl = map(str.strip, pred.strip("() ").split())
                p_name, p_args = spl[0], spl[1:]
                p_objs = []
                for n in p_args:
                    try:
                        p_objs.append(params[n])
                    except KeyError:
                        raise ProblemConfigException("Parameter '%s' for predicate type '%s' not defined in domain file."%(n, p_name))
                init_preds.add(domain.pred_schemas[p_name].pred_class(name="initpred%d"%i,
                                                                      params=p_objs,
                                                                      expected_param_types=domain.pred_schemas[p_name].expected_params))

        # use params and initial preds to create an initial State object
        initial_state = state.State("initstate", params, init_preds, timestep=0)

        # create goal predicate objects
        goal_preds = set()
        for i, pred in enumerate(problem_config["Goal"].split(",")):
            spl = map(str.strip, pred.strip("() ").split())
            p_name, p_args = spl[0], spl[1:]
            p_objs = []
            for n in p_args:
                try:
                    p_objs.append(params[n])
                except KeyError:
                    raise ProblemConfigException("Parameter '%s' for predicate type '%s' not defined in domain file."%(n, p_name))
            goal_preds.add(domain.pred_schemas[p_name].pred_class(name="goalpred%d"%i,
                                                                  params=p_objs,
                                                                  expected_param_types=domain.pred_schemas[p_name].expected_params))

        # use initial state to create Problem object
        initial_problem = problem.Problem(initial_state, goal_preds)
        return initial_problem
