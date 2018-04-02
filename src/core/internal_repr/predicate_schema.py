class PredicateSchema(object):
    """
    A predicate schema holds the following information.

    pred_type: type of this predicate
    pred_class: reference to class for this predicate type
    expected_params: list of expected parameter types as strings
    """
    def __init__(self, pred_type, pred_class, expected_params):
        self.pred_type = pred_type
        self.pred_class = pred_class
        self.expected_params = expected_params
