#!/usr/bin/env python

import argparse
import importlib
import json
import pickle
import sys
import threading
import Queue

import numpy as np

import rospy
from std_msgs.msg import String

import actionlib

import baxter_dataflow
import baxter_interface

from baxter_interface import CHECK_VERSION

from openravepy import Environment

from core.internal_repr.action import Action
from core.internal_repr.parameter import Object, Symbol
from core.internal_repr.plan import Plan
from core.util_classes.robots import Baxter

from baxter_plan import execute_action
from baxter_plan.msg import PlanMSG

class PlanReceiver(object):
	def listen_for_plans(self):
		rospy.init_node('plan_receiver')
		rospy.Subscriber('Plan', PlanMSG, self._execute_plan)
		rospy.spin()

	def _execute_plan(self, data):
			plan = self._build_plan(data)
			pub = rospy.Publisher('Failed_Predicates', String, queue_size=10)
			for action in plan.actions:
				failed_action_preds = action.get_failed_preds()
				if failed_action_preds:
					pub.publish("Failed action {0}. Failed preds: {1}".format(action.name, str(failed_action_preds)))
					# return
				execute_action(action)

	def _build_plan(self, data):
		print "Building plan."
		env = Environment()
		params = {}
		for param in data.parameters:
			new_param = self._build_param(param)
			params[new_param.name] = new_param

		actions = []
		for action in data.actions:
			actions.append(self._build_action(action, params, env))

		return Plan(params, actions, data.horizon, env)


	def _build_action(self, data, plan_params, env):
		params = []
		for param in data.parameters:
			params.append(plan_params[param])

		preds = []
		for pred in data.predicates:
			preds.append(self._build_actionpred(pred, plan_params, env))

		return Action(data.step_num, data.name, data.active_timesteps, params, preds)


	def _build_actionpred(self, data, plan_params, env):
		actionpred = {}
		actionpred['negated'] = data.negated
		actionpred['hl_info'] = data.hl_info
		actionpred['pred'] = self._build_pred(data.predicate, plan_params, env)
		actionpred['active_timesteps'] = (data.active_timesteps[0], data.active_timesteps[1])
		return actionpred


	def _build_pred(self, data, plan_params, env):
		class_path = data.class_path.rsplit(".", 1)
		pred_module = importlib.import_module(class_path[0])
		pred_class = getattr(pred_module, class_path[1])

		params = []
		for param in data.parameters:
			if param.name in plan_params.keys():
				params.append(plan_params[param.name])
			else:
				params.append(self._build_param(param))

		return pred_class(data.name, params, data.param_types, env)


	def _build_param(self, data):
		try:
			param = pickle.loads(data.data)
		except pickle.UnpicklingError:
			print "Could not unpickle parameter."
			raise

		if data.geom.class_path != "":
			param.geom = self._build_geom(data.geom)

		return param


	def _build_geom(self, data):
		class_path = data.class_path.rsplit(".", 1)
		geom_module = importlib.import_module(class_path[0])
		geom_class = getattr(geom_module, class_path[1])
		if data.data:
			try:
				geom = pickle.loads(data.data)
			except pickle.UnpicklingError:
				print "Cannot unpickle geometry {0}.".format(geom_class)
		else:
			geom = geom_class()

		if geom_class is Baxter:
			# Fix this
			geom.shape = "/home/michael/robot_work/tampy/models/baxter/baxter.xml"

		return geom


if __name__ == "__main__":
	PlanReceiver().listen_for_plans()
