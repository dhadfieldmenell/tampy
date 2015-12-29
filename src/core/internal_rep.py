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





