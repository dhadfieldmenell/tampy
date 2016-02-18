from IPython import embed as shell
import importlib
from core.internal_repr import parameter
from core.util_classes import common_predicates
from core.internal_repr.domain import Domain
from core.internal_repr.parameter_schema import ParameterSchema
from core.internal_repr.predicate_schema import PredicateSchema
from core.internal_repr.action_schema import ActionSchema
from errors_exceptions import DomainConfigException, PredicateException, ImpossibleException
import re

class ParseDomainConfig(object):
    """
    Read the domain configuration data and spawn the corresponding Domain object (see Domain class).
    """
    @staticmethod
    def parse(domain_config):
        return Domain(ParseDomainConfig._create_param_schemas(domain_config),
                      ParseDomainConfig._create_pred_schemas(domain_config),
                      ParseDomainConfig._create_action_schemas(domain_config))

    @staticmethod
    def _create_param_schemas(domain_config):
        try:
            attr_paths = domain_config["Attribute Import Paths"]
            attr_paths = dict([l.split() for l in map(str.strip, attr_paths.split(","))])
        except KeyError:
            attr_paths = {}
        for k, v in attr_paths.items():
            attr_paths[k] = importlib.import_module(v)
        param_schemas = {}
        for t in domain_config["Types"].split(";"):
            type_name, attrs = map(str.strip, t.strip(" )").split("("))
            attr_dict = dict([l.split() for l in map(str.strip, attrs.split("."))])
            attr_dict["_type"] = "str"
            assert "name" in attr_dict and ("pose" in attr_dict or "value" in attr_dict)
            for k, v in attr_dict.items():
                if v in attr_paths:
                    if not hasattr(attr_paths[v], v):
                        raise DomainConfigException("%s not found in module %s!"%(v, attr_paths[v]))
                    attr_dict[k] = getattr(attr_paths[v], v)
                else:
                    try:
                        attr_dict[k] = eval(v)
                    except NameError as e:
                        raise DomainConfigException("Need to provide attribute import path for non-primitive %s."%v)
            obj_or_symbol = ParseDomainConfig._dispatch_obj_or_symbol(attr_dict)
            param_schemas[type_name] = ParameterSchema(type_name, getattr(parameter, obj_or_symbol), attr_dict)
        return param_schemas

    @staticmethod
    def _create_pred_schemas(domain_config):
        pred_schemas = {}
        for p_defn in domain_config["Predicates"].split(";"):
            p_type, exp_types = map(str.strip, p_defn.split(",", 1))
            if not hasattr(common_predicates, p_type):
                raise PredicateException("Predicate type '%s' not defined!"%p_type)
            pred_schemas[p_type] = PredicateSchema(p_type, getattr(common_predicates, p_type), [s.strip() for s in exp_types.split(",")])
        return pred_schemas

    @staticmethod
    def _create_action_schemas(domain_config):
        action_schemas = {}
        for k, v in domain_config.items():
            if k.startswith("Action"):
                _, a_name, horizon = k.split()
                # parse out params, predicates, time ranges
                count, inds = 0, [0]
                for i, token in enumerate(v):
                    if token == "(":
                        count += 1
                    if token == ")":
                        count -= 1
                        if count == 0:
                            inds.append(i+1)
                assert len(inds) == 4
                params_str = v[inds[0]:inds[1]].strip()
                pre = v[inds[1]:inds[2]].strip()
                m = re.match("\(\s*and", pre)
                if m:
                    pre = pre[m.span()[1]:-1].strip()
                eff = v[inds[2]:inds[3]].strip()
                m = re.match("\(\s*and", eff)
                if m:
                    eff = eff[m.span()[1]:-1].strip()
                pred_strs = []
                count, prev_i = 0, 0
                p_s = pre + eff
                for i, token in enumerate(p_s):
                    if token == "(":
                        count += 1
                    if token == ")":
                        count -= 1
                        if count == 0:
                            pred_strs.append(p_s[prev_i:i+1].strip())
                            prev_i = i + 1
                all_active_timesteps = v[inds[3]:].strip()
                all_active_timesteps = [tuple(map(int, s.split(":"))) for s in all_active_timesteps.split()]
                # build list of params
                params = []
                for p in params_str.strip("()").split("?"):
                    if p:
                        p_name, p_type = p.strip().split("-")
                        params.append(("?%s"%p_name.strip(), p_type.strip()))
                # build universally quantified params
                univ_params = {}
                for i, pred in enumerate(pred_strs):
                    while True:
                        m = re.match("\(\s*forall", pred)
                        if not m:
                            break
                        pred = pred[m.span()[1]:-1].strip()
                        g = re.match("\((.*?)\)(.*)", pred).groups()
                        loop_var_name, loop_var_type = map(str.strip, g[0].split("-"))
                        pred = g[1].strip()
                        # if this dummy variable name is already used, then change the name
                        if loop_var_name in univ_params:
                            pred = pred.replace(loop_var_name, "%s1"%loop_var_name)
                            loop_var_name = "%s1"%loop_var_name
                        univ_params[loop_var_name] = loop_var_type
                        # replace this predicate in pred_strs because we removed the forall part
                        # (and possibly renamed the dummy variable)
                        pred_strs[i] = pred
                # build preds
                preds = []
                for i, pred in enumerate(pred_strs):
                    # handle not
                    m = re.match("\(\s*not", pred)
                    if m:
                        pred = pred[m.span()[1]:-1].strip()
                    negated = m is not None
                    # parse out predicate type and args
                    spl = pred.strip("() ").split()
                    pred_type, args = spl[0], spl[1:]
                    preds.append({"type": pred_type, "args": args, "negated": negated,
                                  "active_timesteps": all_active_timesteps[i]})
                action_schemas[a_name] = ActionSchema(a_name, int(horizon), params, univ_params, preds)
        return action_schemas

    @staticmethod
    def _dispatch_obj_or_symbol(attr_dict):
        # decide whether this parameter is an Object or Symbol by looking at whether
        # it has an instance attribute named "pose" or one named "value" in the config file
        if "pose" in attr_dict:
            return "Object"
        elif "value" in attr_dict:
            return "Symbol"
        else:
            raise ImpossibleException("Can never reach here.")
