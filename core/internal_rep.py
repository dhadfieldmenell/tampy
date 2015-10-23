"""
basic classes to represent a planning problem. 

Broken down into 
   -- Objects: things in the world, can have a pose
      --> robots are a subclass of objects
     
   -- Predicates: abstract class to represent testable relationships between objects
      
   -- Actions: abstract class to represent actions that can alter the state

   -- State: Collection of predicates and objects
      --> object variables are either fixed or modifiable

   -- Problem: initial state (everything fixed), goal (only properties, nothing fixed)
   -- Plan: sequence of actions
"""

class Obj(object):
    pass

class Robot(Obj):
    pass

class Predicate(object):
    pass

class Action(object):

    def __init__(self, pre, eff, params):
        ## params is a list of objs
        self.params = params
        self.pre = pre
        self.eff = eff

class State(object):
    
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

    def is_abs(self):
        return self._is_abs

class Problem(object):
    
    def __init__(self, init, goal):
        assert goal.is_abs()
        assert init.is_concrete()
        self.init = init
        self.goal = goal        


class Plan(object):
    
    def __init__(self, actions):
        self.actions = actions
