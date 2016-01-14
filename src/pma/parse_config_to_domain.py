from IPython import embed as shell
import hl_solver
import ll_solver
import importlib
from core.internal_repr import parameter
from core.internal_repr import predicate
from core.internal_repr import domain

class ParseConfigToDomain(object):
    """
    Read the domain configuration data and spawn the corresponding Domain object (see Domain class).
    """
    def __init__(self, domain_config):
        self.domain_config = domain_config

    def parse(self):
        # parse out the HLSolver and LLSolver
        if "HLSolver" not in self.domain_config or "LLSolver" not in self.domain_config:
            raise Exception("Must define both HL solver and LL solver in domain config file.")
        s = self.domain_config["HLSolver"]
        if not hasattr(hl_solver, s):
            raise Exception("HLSolver '%s' not defined!"%s)
        hls = getattr(hl_solver, s)()
        s = self.domain_config["LLSolver"]
        if not hasattr(ll_solver, s):
            raise Exception("LLSolver '%s' not defined!"%s)
        lls = getattr(ll_solver, s)()

        # create parameter schema mapping
        attr_paths = self.domain_config["Attribute Import Paths"]
        attr_paths = dict([l.split() for l in map(str.strip, attr_paths.split(","))])
        for k, v in attr_paths.items():
            attr_paths[k] = importlib.import_module(v)
        param_schema = {}
        for t in self.domain_config["Types"].split(";"):
            type_name, attrs = map(str.strip, t.strip(" )").split("("))
            if not hasattr(parameter, type_name):
                raise Exception("Parameter type '%s' not defined!"%type_name)
            attr_dict = dict([l.split() for l in map(str.strip, attrs.split("."))])
            for k, v in attr_dict.items():
                if v in attr_paths:
                    if not hasattr(attr_paths[v], v):
                        raise Exception("%s not found in module %s!"%(v, attr_paths[v]))
                    attr_dict[k] = getattr(attr_paths[v], v)
                else:
                    attr_dict[k] = eval(v)
            param_schema[type_name] = (getattr(parameter, type_name), attr_dict)

        # create predicate schema mapping
        pred_schema = {}
        for p_defn in self.domain_config["Predicates"].split(";"):
            p_name, p_type = map(str.strip, p_defn.split(",", 1))
            if not hasattr(predicate, p_name):
                raise Exception("Predicate type '%s' not defined!"%p_name)
            pred_schema[p_name] = (getattr(predicate, p_name), [s.strip() for s in p_type.split(",")])

        # create action schema mapping
        action_schema = {}

        return domain.Domain(hls, lls, hls.translate_domain(self.domain_config), param_schema, pred_schema, action_schema)
