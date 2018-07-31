from pma import backtrack_ll_solver

class NAMOSolver(backtrack_ll_solver.BacktrackLLSolver):
    def get_rs_param(self, a):
        if a.name == 'moveto':
            ## find possible values for the final pose
            rs_param = a.params[2]
        elif a.name == 'movetoholding':
            ## find possible values for the final pose
            rs_param = a.params[2]
        elif a.name == 'grasp':
            ## sample the grasp/grasp_pose
            rs_param = a.params[4]
        elif a.name == 'putdown':
            ## sample the end pose
            rs_param = a.params[4]
        else:
            raise NotImplemented

        return rs_param
