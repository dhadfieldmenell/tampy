"""
The state is represented by a collection of predicates and objects.
Object variables are either fixed or modifiable.
"""

class State:
    def __init__(self, objs, preds):
        self.objs = objs
        self.preds = preds
        self.consistent = True
        for p in self.preds:
            if not p.test(objs):
                self.consistent = False
        self._is_abs = any([o.is_var() for o in objs])

    def is_concrete(self):
        return not self._is_abs

    def is_abstract(self):
        return self._is_abs
