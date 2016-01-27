class ProblemConfigException(Exception):
    """ Either config not found or config format incorrect """
    pass

class DomainConfigException(Exception):
    """ Either config not found or config format incorrect """
    pass

class SolversConfigException(Exception):
    """ Either config not found or config format incorrect """
    pass

class ParamValidationException(Exception):
    """ Check validate_params functions """
    pass

class PredicateException(Exception):
    """ Predicate type mismatch, not defined, or parameter range violation """
    pass

class HLException(Exception):
    """ An issue with the high level solver (hl_solver) """
    pass

class LLException(Exception):
    """ An issue with the low level solver (ll_solver) """
    pass

class ImpossibleException(Exception):
    """ This exception should never be raised """
    pass