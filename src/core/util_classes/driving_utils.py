import numpy as np

from driving_sim.driving_gui.gui import GUI
from driving_sim.internal_state.objects.object import DrivingObject
from driving_sim.internal_state.surfaces.surface import DrivingSurface

def get_driving_gui(plan):
    plan.env.update_horizon(horizon)
    return GUI(plan.env)

def transfer_to_sim(plan, param_name):
    param = plan.params[param_name]
    if param.is_symbol(): raise ValueError("Cannot transfer symbol {} into the simulator.".format(param.name))
    if not hasattr(param, 'geom') or param.geom is None: raise ValueError("Cannot transfer {} into the simulator without a geom attribute.".format(param.name))

    if isinstance(param.geom, DrivingObject):
        for attr in dir(param):
            param_attr = getattr(param, attr)
            if isinstance(param_attr, np.ndarray) and hasattr(param.geom, attr):
                getattr(param.geom, attr)[:] = param_attr.flatten()
            elif attr == 'xy':
                # Need special handling on this one since it's split into two values in the simulator
                param.geom.x[:], param.geom.y[:] = param.xy[0,:], param.xy[1,:]


def transfer_plan_to_sim(plan):
    plan.env.update_horizon(plan.horizon)
    for param_name in plan.params:
        if plan.params[param_name].is_symbol(): continue
        transfer_to_sim(plan, param_name)
