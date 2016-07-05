class TampyException(Exception):
    """ A generic exception for Tampy """

class ProblemConfigException(TampyException):
    """ Either config not found or config format incorrect """
    pass

class DomainConfigException(TampyException):
    """ Either config not found or config format incorrect """
    pass

class SolversConfigException(TampyException):
    """ Either config not found or config format incorrect """
    pass

class ParamValidationException(TampyException):
    """ Check validate_params functions """
    pass

class PredicateException(TampyException):
    """ Predicate type mismatch, not defined, or parameter range violation """
    pass

class HLException(TampyException):
    """ An issue with the high level solver (hl_solver) """
    pass

class LLException(TampyException):
    """ An issue with the low level solver (ll_solver) """
    pass

class OpenRAVEException(TampyException):
    """ An OpenRAVE related issue"""
    pass
class ImpossibleException(TampyException):

    """ This exception should never be raised """
    pass
