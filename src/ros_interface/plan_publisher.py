import numpy as np
import pickle

import rospy
from std_msgs.msg import String

from ros_interface.msg._ActionMSG import ActionMSG
from ros_interface.msg._ActionPredMSG import ActionPredMSG
from ros_interface.msg._FloatArrayMSG import FloatArrayMSG
from ros_interface.msg._GeomMSG import GeomMSG
from ros_interface.msg._ParameterMSG import ParameterMSG
from ros_interface.msg._PlanMSG import PlanMSG
from ros_interface.msg._PredicateMSG import PredicateMSG

class PlanPublisher(object):
	def publish_plan(self, plan):
		pub = rospy.Publisher('Plan', PlanMSG, queue_size=10)
		rospy.init_node('planner', anonymous=True)
		msg = self.create_plan_msg(plan)
		pub.publish(msg)


	def create_geom_msg(self, geom):
		geom_msg = GeomMSG()
		geom_msg.class_path = str(type(geom)).split("'")[1]

		try:
			geom_msg.data = pickle.dumps(geom)
		except pickle.PicklingError:
			print "Could not pickle {0}.".format(type(geom))
			geom_msg.data = pickle.dumps(None)

		return geom_msg


	def create_parameter_msg(self, param):
		param_msg = ParameterMSG()

		param_msg.name = param.name

		geom = None
		if hasattr(param, 'geom'):
			geom = param.geom
			param.geom = None
			param_msg.geom = self.create_geom_msg(geom)

		if hasattr(param, 'openrave_body'):
			param.openrave_body = None

		try:
			param_msg.data = pickle.dumps(param)
		except pickle.PicklingError:
			print "Could not pickle {0}.".format(param)

		if geom:
			param.geom = geom

		return param_msg


	def create_predicate_msg(self, pred):
		pred_msg = PredicateMSG()
		pred_msg.class_path = str(type(pred)).split("'")[1]
		params = pred.params
		env = pred.env

		pred_msg.name = pred.name
		for param in pred.params:
			pred_msg.parameters.append(self.create_parameter_msg(param))
			pred_msg.param_types.append(param.get_type())

		return pred_msg


	def create_actionpred_msg(self, actionpred):
		actionpred_msg = ActionPredMSG()
		actionpred_msg.negated = actionpred['negated']
		actionpred_msg.hl_info = actionpred['hl_info']
		actionpred_msg.predicate = self.create_predicate_msg(actionpred['pred'])
		actionpred_msg.active_timesteps = actionpred['active_timesteps']
		return actionpred_msg


	def create_action_msg(self, action):
		action_msg = ActionMSG()
		action_msg.name = action.name
		action_msg.active_timesteps = action.active_timesteps
		action_msg.step_num = action.step_num
		for param in action.params:
			action_msg.parameters.append(param.name)
		for pred in action.preds:
			action_msg.predicates.append(self.create_actionpred_msg(pred))
		return action_msg


	def create_plan_msg(self, plan):
		plan_msg = PlanMSG()
		plan_msg.horizon = plan.horizon
		for parameter in plan.params.values():
			plan_msg.parameters.append(self.create_parameter_msg(parameter))
		for action in plan.actions:
			plan_msg.actions.append(self.create_action_msg(action))

		return plan_msg


if __name__ == "__main__":
	pb = PlanPublisher()
