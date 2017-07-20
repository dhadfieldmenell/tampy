import numpy as np

"""
Truncated Image Range
x: 98-288
y: 65-260
"""

class ClothNet(object):
	def __init__(self):
		self.ref_images = np.load('ros_interface/cloth/data/truncatedImages.npy')
		self.ref_labels = np.load('ros_interface/cloth/data/truncatedLabels.npy')

	def build(self):
		pass

	def predict(self, image):
		best_dist = 1e20
		cur_lab = None
		for i in range(len(self.ref_images)):
			ref_im = self.ref_images[i]
			dist = np.sum((ref_im - image)**2)
			if dist < best_dist:
				best_dist = dist
				cur_lab = self.ref_labels[i]
		return [cur_lab.copy()]
