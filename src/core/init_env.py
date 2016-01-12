from IPython import embed as shell
import numpy as np

class InitEnv:
    """
    Read the domain-specific environment file(s) and use it to spawn relevant data. Also, use
    it to set the parameter trajectory tables for HL search nodes (which works because these tables
    only hold the first timestep of data). This method is used by parse_config_to_problem.
    """
    def construct_env_and_init_params(self, env_file, params, preds):
        raise NotImplementedError("Override this.")

class InitNAMOEnv(InitEnv):
    def construct_env_and_init_params(self, env_file, params, preds):
        # create grid map
        grid = []
        with open("../environments/%s"%env_file, "r") as f:
            for line in f:
                grid.append(list(line.strip()))
        grid = np.array(grid)
        h, w = grid.shape

        # initialize target and robot locations
        def is_int(s):
            try:
                int(s)
                return True
            except ValueError:
                return False
        for i in range(h):
            for j in range(w):
                if is_int(grid[i, j]):
                    params["target%s"%grid[i, j]].pose = np.array([i, j]).reshape(2, 1)
                elif grid[i, j] == "R":
                    params["pr2"].pose = np.array([i, j]).reshape(2, 1)

        # initialize can locations to start at target, as defined by At predicates
        for pred in preds:
            if pred.get_type() == "At":
                pred.params[0].pose = pred.params[1].pose

        env_data = {"w": w, "h": h}
        return env_data

class InitCanEnv(InitEnv):
    def construct_env_and_init_params(self, env_file, params, preds):
        # TODO: read file to spawn things like table
        env_data = {}
        params["pr2"].pose = 0
        params["can1"].pose = 0
        params["can2"].pose = 0
        params["can3"].pose = 0
        params["lgripper"].pose = 0
        params["rgripper"].pose = 0
        return env_data
