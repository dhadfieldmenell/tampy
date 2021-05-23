from core.internal_repr import state
from core.internal_repr import problem
from errors_exceptions import ProblemConfigException

from driving_sim.internal_state.simulator_state import SimulatorState

def update_or_spawn_sim(env, params):
    if env is None:
        env = SimulatorState(2, params['x_limit'].value[0,0], params['y_limit'].value[0, 0])

    env.clear()

    for param in params:
        if hasattr(param, 'geom'):
            env.add(param._type, param.geom)

    return env

class ParseDrivingProblemConfig(object):
    """
    Read the problem configuration data and spawn the corresponding initial Problem object (see Problem class).
    This is only done for spawning the very first Problem object, from the initial state specified in the problem configuration file.
    Validation is performed against the schemas stored in the Domain object self.domain.
    """
    @staticmethod
    def parse(problem_config, domain, env=None):
        # create parameter objects
        params = {}

        if "Objects" not in problem_config or not problem_config["Objects"]:
            raise ProblemConfigException("Problem file needs objects.")
        for t in problem_config["Objects"].split(";"):
            if t.strip() == '': continue
            o_type, attrs = list(map(str.strip, t.strip(" )").split("(", 1)))
            attr_dict = {}
            for l in map(str.strip, attrs.split(".")):
                lst = l.split(" ", 1)
                k, v = lst[0], lst[1:]
                attr_dict[k] = v
            assert "name" in attr_dict
            name = attr_dict["name"][0]
            attr_dict["_type"] = [o_type]
            params[name] = attr_dict

        if "Init" not in problem_config or not problem_config["Init"]:
            raise ProblemConfigException("Problem file needs init.")
        prim_preds, deriv_preds = list(map(str.strip, problem_config["Init"].split(";")))
        if prim_preds:
            for pred in map(str.strip, prim_preds.split(")")):
                if pred:
                    a, b = pred.find("["), pred.rfind("]") + 1
                    if a != -1:
                        new_s = "".join(pred[a:b].split())
                        pred = pred.replace(pred[a:b], new_s)
                    lst = list(map(str.strip, pred.strip(",() ").split()))
                    k = lst[0]
                    obj_name = lst[1]
                    v = lst[2:]
                    if obj_name not in params:
                        raise ProblemConfigException("'%s' is not an object in problem file."%obj_name)
                    params[obj_name][k] = [x.replace("[", "(").replace("]", ")") for x in v]
            for obj_name, attr_dict in list(params.items()):
                o_type = attr_dict["_type"][0]
                name = attr_dict["name"][0]
                if 'geom' in attr_dict:
                    attr_dict['geom'] = list(map(eval, attr_dict['geom']))

                try:
                    params[obj_name] = domain.param_schemas[o_type].param_class(attrs=attr_dict,
                                                                                attr_types=domain.param_schemas[o_type].attr_dict)

                except KeyError:
                    import ipdb; ipdb.set_trace()
                    raise ProblemConfigException("Parameter '%s' not defined in domain file."%name)
                except ValueError:
                    raise ProblemConfigException("Some attribute type in parameter '%s' is incorrect."%name)
        for k, v in list(params.items()):
            if type(v) is dict:
                raise ProblemConfigException("Problem file has no primitive predicates for object '%s'."%k)

        env = update_or_spawn_sim(env, params)

        init_preds = set()
        if deriv_preds:
            for i, pred in enumerate(deriv_preds.split(",")):
                spl = list(map(str.strip, pred.strip("() ").split()))
                p_name, p_args = spl[0], spl[1:]
                p_objs = []
                for n in p_args:
                    try:
                        p_objs.append(params[n])
                    except KeyError:
                        raise ProblemConfigException("Parameter '%s' for predicate type '%s' not defined in domain file."%(n, p_name))
                try:
                    init_preds.add(domain.pred_schemas[p_name].pred_class(name="initpred%d"%i,
                                                                          params=p_objs,
                                                                          expected_param_types=domain.pred_schemas[p_name].expected_params,
                                                                          env=env))
                except TypeError:
                    print(("type error for {}".format(pred)))

        # use params and initial preds to create an initial State object
        initial_state = state.State("initstate", params, init_preds, timestep=0)

        # create goal predicate objects
        goal_preds = set()
        for i, pred in enumerate(problem_config["Goal"].split(",")):
            spl = list(map(str.strip, pred.strip("() ").split()))
            p_name, p_args = spl[0], spl[1:]
            p_objs = []
            for n in p_args:
                try:
                    p_objs.append(params[n])
                except KeyError:
                    raise ProblemConfigException("Parameter '%s' for predicate type '%s' not defined in domain file."%(n, p_name))
            goal_preds.add(domain.pred_schemas[p_name].pred_class(name="goalpred%d"%i,
                                                                  params=p_objs,
                                                                  expected_param_types=domain.pred_schemas[p_name].expected_params, env=env))

        # use initial state to create Problem object
        initial_problem = problem.Problem(initial_state, goal_preds, env)
        return initial_problem
