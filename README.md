# TAMPy
This README assumes the installation is happenning in ubuntu >= 14 with python3.5


## Setup

In order to run this code, first run `source setup.sh`. If you wish to only use TAMP functionality and none of the in-development learning components, edit `setup.sh` so that `FULL_INSTALL=false`
This will create a directory titled `tamp_work` in your home directory and initialize a python virtual environment with the necessary dependencies. To activate this environment, you can simply run `tampenv`.


## Verify planning

Once installed, navigate to `cd ~/tamp_work/tampy/src` and run `python3 verify_namo.py`
This will solve a two-object placement problem in a two dimensional domain
If everything works, the output should match: `PLAN FINISHED WITH FAILED PREDIATES:[]`

This should take under a minute to run.


## The code

### Defining domains

#### Domain files
Specifications for all domains should be placed in the `domains` folder directly under the `tampy` directory

For a concrete example, refer to `domains/namo_domain/generate_namo_domain.py`. This is script designed to generate a domain file (these files are cumbersome to write directly by hand)

##### Types
The first portion of the file will look like

`Types: Can, Target, RobotPose, Robot, Grasp, Obstacle`

Here, everything following `Types:` is a "type" parameters in the domain can be

##### Import Paths
The next portion specifies where to find various necessary code

First:

`Attribute Import Paths: RedCircle core.util_classes.items, Vector1d core.util_classes.matrix, Vector2d core.util_classes.matrix, Wall core.util_classes.items, NAMO core.util_classes.robots`

Here, we tell the planner where to find the python classes parameter attributes can take on (used in Primitive Predicates)

Second:

`Predicates Import Path: core.util_classes.namo_predicates`

Here, we tell the planner where to find the python classes defining predicates

##### Listing predicates
The next portion specifies primitive and derived predicates

First:

`Primitive Predicates: geom, Can, RedCircle; pose, Can, Vector2d` etc...

This defines a list of 3-tuples of the form `(attribute_name, parameter_type, attribute_type)` and effectively tells the planner how to build parameters

Second:

`Derived Predicates: At, Can, Target; RobotAt, Robot, RobotPose; InGripper, Robot, Can, Grasp; Obstructs, Robot, Target, Target, Can`

This defines a list of tuples whose first element is a predicate class and the remaining are the type signatures for parameters of the predicate. In the above, the `At` predicate is defined to take both a `Can` object and a `Target` symbol

##### Defining actions
An action has five components: a name, a number of timesteps, a list of typed arguments, a "pre" list, and a "post" list
In the generate file above, these are defined as attributes of an `Action` class

Suppose we have a moveto action.

For args:
`args = '(?robot - Robot ?can - Can ?target - Target ?sp - RobotPose ?gp - RobotPose ?g - Grasp)'`

Means the action takes in a robot, a can, a target, a start pose, a goal pose, and a grasp

The `pre` list contains both precondition and midconditions for the action (originally, everything here was preconditions but that was impractical)

Preconditions are identified by being active at `0:0` i.e. only the first timestep

Items of this list are pairs of strings, the first describing the constraint and the second specifying what timesteps to enforce that constraint on

`('(At ?can ?target)', '0:0')` specifies that a precondition should be enforced constraining `?can` to be at `?target`

`('(forall (?w - Obstacle) (not (RCollides ?robot ?w)))', '0:{0}'.format(end))` specifies that for all objects of type `Obstacle` (not just those in the action arguments), the action should include a constraint prohibiting collision between the object and the robot from time 0 to time `end`

Note the syntax: `forall` allows ranging over the global space of parameters, not just those immediatley visible to the action, while `not` enforces the negation of a constraint (here, `RCollides` is a constraint to be in collision, so adding the `not` instead contrains to avoid collision)


The `eff` list is similar, but everything in this list will be considered a postcondition by the task planner and is something that must be true when the action finishes.

`('(forall (?obj - Can) (Stationary ?obj))', '{0}:{1}'.format(end, end-1))`

Note in the above how the timesteps are `(end, end-1)`; this is special syntax telling the motion planner to ignore the constraint even though the task planner will include it, and allows adding extra information to guide the task planner that is uncessary for the motion planner

#### Problem files
Specifications for all problem files should be placed under the relevant domain directory 

For a concrete example, refer to `domains/namo_domain/generate_namo_prob.py`. This is script designed to generate a problem file (these files are cumbersome to write directly by hand)

A problem file will define a list of objects, initial attribute values for those objects, a goal, and a set of initial conditions

The first part is of the form `Objects: ...`; everything on this line is of the form `ObjectType (name {insert name})` and will list EVERY object and symbol the planner will have access to. Semicolons delimit separate objects and the end of this line must be two new lines `\n\n`

The next part is of the form `Init: ...`; everything in this part is of the form `(attribute objectname value)` and specifies the initial value of every attribute for every object.

`(pose can0 [0,0])` for example specifies that the initial pose of `can0` will be at `[0,0]`

`(pose can1 undefined)` on the other hand specifies that the initial pose of `can1` is not fixed and the planner will determine its value. Items here are delimited by commas and the end of this part must be a semicolon.

The next part is then a list of predicates; these are constraints that ARE true at the initial timesteps. These must be satisfied by the initial values from the preceding part. This part is ended with two new lines `\n\n`

Finally, the last line is of the form `GOAL: ...` and specifies what constraints you want to be true at the end of the planning process.

During planning, the list of initial predicates will be converted to the initial state in PDDL while the goal will be converted likewise

#### Predicates
Predicates will define both constraints for the trajectory optimization problem as well as PDDL-style predicates for the task planner

All predicates will be subclasses of `ExprPredicate` from `core.util_classes.common_predicates`

There are roughly two types of predicates: linear and non-linear

Each predicate has a priority which can be interpreted as order-to-add: the solver will iteratively optimize problems with constraints restricted up-to the current priority, only adding the next priority when the current problem is solved. Usually, higher priority means a harder to solve constraint like collision avoidance

An example linear predicate: 
```
class At(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        self.can, self.targ = params
        attr_inds = OrderedDict([(self.can, [("pose", np.array([0,1], dtype=np.int))]),
                                 (self.targ, [("value", np.array([0,1], dtype=np.int))])])

        A = np.c_[np.eye(2), -np.eye(2)]
        b = np.zeros((2, 1))
        val = np.zeros((2, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types, priority=-2)
```
What's going on here:
- `attr_inds` is a dictionary describing which indices of which atrributes will be used from each parameter; e.g. indices 0 and 1 of the can's pose will be included in the state vector x
- `A` defines a matrix such that Ax = [0., 0.] iff. x[:2] == x[2:]; in this context that means can.pose == targ.value
- `aff_e` is an affine expression of the form `Ax+b`
- `e` is then an affine constraint of the form `Ax+b=val`

Predicates can be defined over multiple timesteps

An example multi-timestep predicate:
```
class RobotStationary(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.c,  = params
        attr_inds = OrderedDict([(self.c, [("pose", np.array([0, 1], dtype=np.int))])])
        A = np.array([[1, 0, -1, 0],
                      [0, 1, 0, -1]])
        b = np.zeros((2, 1))
        e = EqExpr(AffExpr(A, b), np.zeros((2, 1)))
        super(RobotStationary, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), priority=-2)
```
What's going on here:
- `active_range` implies the state vector will be constructed using both the current timestep and the next; generally an `active_range` of the form `(a,b)` implies the state vector will use every timestep from `cur_ts+a` to `cur_ts+b`; using negative values to specifiy earlier timesteps


An example non-linear predicate
```
class InGraspAngle(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        self.r, self.can = params
        self.dist = 0.6
        attr_inds = OrderedDict([(self.r, [("pose", np.array([0, 1], dtype=np.int)),
                                           ("theta", np.array([0], dtype=np.int))]),
                                 (self.can, [("pose", np.array([0, 1], dtype=np.int))]),
                                ])

        def f(x):
            targ_loc = [-self.dist * np.sin(x[2]), self.dist * np.cos(x[2])]
            can_loc = x[3:5]
            return np.array([[((x[0]+targ_loc[0])-can_loc[0])**2 + ((x[1]+targ_loc[1])-can_loc[1])**2]])

        def grad(x):
            curdisp = x[3:5] - x[:2]
            theta = x[2]
            targ_disp = [-self.dist * np.sin(theta), self.dist * np.cos(theta)]
            off = (curdisp[0]-targ_disp[0])**2 + (curdisp[1]-targ_disp[1])**2
            (x1, y1), (x2, y2) = x[:2], x[3:5]

            x1_grad = -2 * ((x2-x1)+self.dist*np.sin(theta))
            y1_grad = -2 * ((y2-y1)-self.dist*np.cos(theta))
            theta_grad = 2 * dist * ((x2-x1)*np.cos(theta) + (y2-y1)*np.sin(theta))
            x2_grad = 2 * ((x2-x1)+self.dist*np.sin(theta))
            y2_grad = 2 * ((y2-y1)-self.dist*np.cos(theta))
            return np.array([x1_grad, y1_grad, theta_grad, x2_grad, y2_grad]).reshape((1,5))

        self.f = f
        self.grad = grad
        angle_expr = Expr(f, grad)
        e = EqExpr(angle_expr, np.zeros((1,1)))
        super(InGraspAngle, self).__init__(name, e, attr_inds, params, expected_param_types, priority=1)
```
What's going on here:
- This is a constraint specifiying that `can` must be at distance `0.6` along direction `theta` from `r`
- `f` replaces `aff_expr` from above with a black-box function call; `f` can only take the state vector as input and can only return a 1-d array
- `grad` will return the jacobian of `f` with respect to the state vector, or more precisely an array of the gradients of each output of `f`. `grad` can only return a 2-d array of shape `(len(f(x)), len(x))` such that `grad(x).dot(x) + f(x)` is valid
- `e` is an equality constraint of the form `f(x)=0`; `angle_expr` provides both `f` and it's gradient `grad` to the solver


