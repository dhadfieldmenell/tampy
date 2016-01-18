from IPython import embed as shell
import importlib
from core.internal_repr import parameter
from core.internal_repr import predicate
from core.internal_repr import domain

class ParseDomainConfig(object):
    """
    Read the domain configuration data and spawn the corresponding Domain object (see Domain class).
    """
    def __init__(self, domain_config):
        self.domain_config = domain_config

    def parse(self):
        # create parameter schema mapping
        attr_paths = self.domain_config["Attribute Import Paths"]
        attr_paths = dict([l.split() for l in map(str.strip, attr_paths.split(","))])
        for k, v in attr_paths.items():
            attr_paths[k] = importlib.import_module(v)
        param_schema = {}
        for t in self.domain_config["Types"].split(";"):
            type_name, attrs = map(str.strip, t.strip(" )").split("("))
            attr_dict = dict([l.split() for l in map(str.strip, attrs.split("."))])
            attr_dict["_type"] = "str"
            assert "name" in attr_dict and ("pose" in attr_dict or "value" in attr_dict)
            for k, v in attr_dict.items():
                if v in attr_paths:
                    if not hasattr(attr_paths[v], v):
                        raise Exception("%s not found in module %s!"%(v, attr_paths[v]))
                    attr_dict[k] = getattr(attr_paths[v], v)
                else:
                    attr_dict[k] = eval(v)
            obj_or_symbol = self._dispatch_obj_or_symbol(attr_dict)
            param_schema[type_name] = (getattr(parameter, obj_or_symbol), attr_dict)

        # create predicate schema mapping
        pred_schema = {}
        for p_defn in self.domain_config["Predicates"].split(";"):
            p_name, p_type = map(str.strip, p_defn.split(",", 1))
            if not hasattr(predicate, p_name):
                raise Exception("Predicate type '%s' not defined!"%p_name)
            pred_schema[p_name] = (getattr(predicate, p_name), [s.strip() for s in p_type.split(",")])

        # create action schema mapping
        action_schema = {}

        return domain.Domain(param_schema, pred_schema, action_schema)

    def _dispatch_obj_or_symbol(self, attr_dict):
        # decide whether this parameter is an Object or Symbol by looking at whether
        # it has an instance attribute named "pose" or one named "value" in the config file
        if "pose" in attr_dict:
            return "Object"
        elif "value" in attr_dict:
            return "Symbol"
        else:
            raise Exception("Can never reach here.")
