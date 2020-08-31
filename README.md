# TAMPy
This README assumes the installation is happenning in ubuntu >= 14 with python3.5


## Setup

In order to run this code, first run `source setup.sh`. If you wish to only use TAMP functionality and none of the in-development learning components, edit `setup.sh` so that `FULL_INSTALL=false`
This will create a directory titled 'tamp_work' in your home directory and initialize a python virtual environment with the necessary dependencies. To activate this environment, you can simply run `tampenv`.


## Verify planning

Once installed, navigate to `cd ~/tamp_work/tampy/src` and run `python verify_namo.py`
This will solve a two-object placement problem in a two dimensional domain
If everything works, the output should match: `PLAN FINISHED WITH FAILED PREDIATES:[]`

This should take under a minute to run.


## The code

### Defining domains

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
What;s going on here:
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


