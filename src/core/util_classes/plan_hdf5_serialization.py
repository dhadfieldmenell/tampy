import h5py
import importlib
import os
import pickle

from openravepy import Environment

from core.internal_repr.action import Action
from core.internal_repr.parameter import Object, Symbol
from core.internal_repr.plan import Plan
from core.util_classes.robots import Baxter, PR2

class PlanSerializer:
    def write_plan_to_hdf5(self, file_name, plan):
        if file_name[-5:] != '.hdf5':
            file_name += '.hdf5'

        hdf5_file = h5py.File(file_name, 'w')
        plan_group = hdf5_file.create_group('plan')
        plan_group['horizon'] = plan.horizon
        action_group = plan_group.create_group('actions')
        param_group = plan_group.create_group('params')

        for param in plan.params.values():
            self._add_param_to_group(param_group, param)
        
        for action in plan.actions:
            self._add_action_to_group(action_group, action)


    def _add_action_to_group(self, group, action):
        action_group = group.create_group(action.name)
        action_group['name'] = action.name
        action_group['active_ts'] = action.active_timesteps
        action_group['params'] = map(lambda p: p.name, action.params)
        action_group['step_num'] = action.step_num
        pred_group = action_group.create_group('preds')

        for i in range(len(action.preds)):
            pred = action.preds[i]
            self._add_action_pred_to_group(pred_group, pred, i)


    def _add_action_pred_to_group(self, group, pred, index):
        pred_group = group.create_group(str(index))
        pred_group['negated'] = pred['negated']
        pred_group['hl_info'] = pred['hl_info']
        pred_group['active_ts'] = pred['active_timesteps']
        self._add_pred_to_group(pred_group, pred['pred'])


    def _add_pred_to_group(self, group, pred):
        pred_group = group.create_group('pred')
        pred_group['class_path'] = str(type(pred)).split("'")[1]
        pred_group['name'] = pred.name
        # Note: Assumes parameter names and types will be at most 64 characters long
        param_dset = pred_group.create_dataset('params', (len(pred.params),), dtype='S64')
        param_types_dset = pred_group.create_dataset('param_types', (len(pred.params),), dtype='S64')
        for i in range(len(pred.params)):
            param_dset[i] = pred.params[i].name
            param_types_dset[i] = pred.params[i].get_type()


    def _add_param_to_group(self, group, param):
        param_group = group.create_group(param.name)
        param_group['name'] = param.name

        geom = None
        if hasattr(param, 'geom') and param.geom:
            geom = param.geom
            param.geom = None
            self._add_geom_to_group(param_group, geom)

        or_body = None
        if hasattr(param, 'openrave_body'):
            or_body = param.openrave_body
            param.openrave_body = None

        try:
            param_group['data'] = pickle.dumps(param)
        except pickle.PicklingError:
            print "Could not pickle {0}.".format(param.name)

        if geom:
            param.geom = geom

        if or_body:
            param.openrave_body = or_body


    def _add_geom_to_group(self, group, geom):
        geom_group = group.create_group('geom')
        geom_group['class_path'] = str(type(geom)).split("'")[1]
        try: 
            geom_group['data'] = pickle.dumps(geom)
        except pickle.PicklingError:
            print "Could not pickle {0}.".format(type(geom))
            geom_group['data'] = pickle.dumps(None)
            

class PlanDeserializer:
    def read_from_hdf5(self, file_name):
        try:
            file = h5py.File(file_name, 'r')
        except IOError:
            print 'Cannot read plan from hdf5: No such file or directory.'
            return

        if 'plan' not in file:
            print 'Cannot read plan from hdf5: File does not contain a plan.'
            return

        return self._build_plan(file['plan'])


    def _build_plan(self, group):
        env = Environment()
        params = {}
        for param in group['params'].values():
            new_param = self._build_param(param)
            params[new_param.name] = new_param

        actions = []
        for action in group['actions'].values():
            actions.append(self._build_action(action, params, env))

        return Plan(params, actions, group['horizon'].value, env, determine_free=False)


    def _build_action(self, group, plan_params, env):
        params = []
        for param in group['params']:
            params.append(plan_params[param])

        preds = []
        for pred in group['preds'].values():
            preds.append(self._build_actionpred(pred, plan_params, env))

        active_timesteps = (group['active_ts'].value[0], group['active_ts'].value[1])

        return Action(group['step_num'].value, group['name'].value, active_timesteps, params, preds)


    def _build_actionpred(self, group, plan_params, env):
        actionpred = {}
        actionpred['negated'] = group['negated'].value
        actionpred['hl_info'] = group['hl_info'].value
        actionpred['pred'] = self._build_pred(group['pred'], plan_params, env)
        actionpred['active_timesteps'] = (group['active_ts'].value[0], group['active_ts'].value[1])
        return actionpred


    def _build_pred(self, group, plan_params, env):
        class_path = group['class_path'].value.rsplit(".", 1)
        pred_module = importlib.import_module(class_path[0])
        pred_class = getattr(pred_module, class_path[1])

        params = []
        for param in group['params']:
            if param in plan_params:
                params.append(plan_params[param])
            else:
                print 'Param {0} for pred {1} was not serialized with plan.'.format(param, class_path)

        return pred_class(group['name'].value, params, group['param_types'].value, env)


    def _build_param(self, group):
        try:
            param = pickle.loads(group['data'].value)
        except pickle.UnpicklingError:
            print "Could not unpickle parameter."
            raise

        if 'geom' in group:
            param.geom = self._build_geom(group['geom'])

        return param


    def _build_geom(self, group):
        class_path = group['class_path'].value.rsplit(".", 1)
        geom_module = importlib.import_module(class_path[0])
        geom_class = getattr(geom_module, class_path[1])
        if 'data' in group.keys():
            try:
                geom = pickle.loads(group['data'].value)
            except pickle.UnpicklingError:
                print "Cannot unpickle geometry {0}.".format(geom_class)
        else:
            geom = geom_class()

        if geom_class is Baxter:
            base_dir = os.getcwd()
            geom.shape = "{0}/../models/baxter/baxter.xml".format(base_dir)
        
        if geom_class is PR2:
            base_dir = os.getcwd()
            geom.shape = "{0}/../models/pr2/pr2.zae".format(base_dir)
                
        return geom
