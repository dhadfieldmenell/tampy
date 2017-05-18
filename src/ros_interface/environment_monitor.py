import copy

import rospy

class EnvironmentMonitor:
	def __init__(self, plan_params, env):
		self._plan_params = plan_params
		self._env = env
		self._recognized_objects


	def listen_to_env(data):
		rospy.Subscriber('Environment', RecognizedObject, self.update_objects)


	def update_objects(self, data):
		# if not data.pose or not data.rotation or not data.type:
		# 	raise Exception('Could not parse recognized object')

		self._recognized_objects = data


	def update_params(self, t):
		self._match_objects()
		obj = self.find_object(data)
		if not obj:
			self.add_to_env(data)
		else:
			self.update_param(obj, data)


	def _match_objects(self):
		matchings = {}
		rec_objects = copy.copy(self._recognized_objects)

		for obj in rec_objects:
			params = copy.copy(self._plan_params)
			matching_index, dist = self._find_object_index_dist(obj, params)
			matching = params[matching_index]

			while matching && matching in matchings:
				dist0 = self.find_distance(obj.pose, matchings[matching].pose)
				if dist0 < dist:
					del params[i]
					matching_index = self._find_object_index_dist(obj, params)
					matching = params[matching_index]
				else:
					rec_objects.append(matchings[matching])

			if matching:
				matchings[matching] = obj

		return matchings


	def _find_object_index_dist(self, obj, params):
		matching = 0
		min_distance = -1

		for i in range(len(params)):
			param = params[i]
			if obj.type != param.type:
				continue

			dist = self.find_distance(obj.pose, param.pose)
			if dist < min_distance or min_distance < 0:
				min_distance = dist
				matching = i

		return matching, min_distance


	def add_param_to_env(self, param):
		pass


	def update_param(self, obj, data):
		obj.pose = self._extract_position(data.position)
		obj.rotation = self._extract_rotation(data.quaternion)


	def _extract_position(self, position):
		return [position.x, position.y, position.z]


	def _extract_rotation(self, quaternion):
		quaternion = [quaternion.x, quaternion.y, quaternion,z, quaternion.w]
		rotation = tf.transformations.euler_from_quaternion(quaternion)
		return rotation
