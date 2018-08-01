import numpy as np
# Assumptions hardcoded
# Assume picture of 3 x 160 x 120, Depth x Height x Width
x_lim = (-3, 10)
y_lim = (-5, 10)
top_left = (x_lim[0], y_lim[1])
scaling_x = float(x_lim[1] - x_lim[0]) / 120
scaling_y = float(y_lim[1] - y_lim[0]) / 160
pic_size = 10

def img_partition_collector_existence(img, label, start_coord=(0,0)):
	"""
	img is assumed to be 3 x 160 x 120, Depth x Height x Width
	label is assumed to be x,y coordinate of robot center
	start_coord is in pixels
	pic_size is in pixels
	"""
	# Calculate the center of the picture relative to pixels
	center = (label[0], label[1]) # This is in real units ==> Convert to pixels
	center = ((center[0] - top_left[0])/scaling_x, (top_left[1] - center[1])/scaling_y)
	split_imgs = []
	split_imgs_labels = []
	num_iter_x = (120 - start_coord[0])/pic_size
	num_iter_y = (160 - start_coord[1])/pic_size
	for x in range(num_iter_x): 
		for y in range(num_iter_y): 
			# import ipdb; ipdb.set_trace()
			start_x = start_coord[0] + pic_size * x
			start_y = start_coord[1] + pic_size * y
			print(start_x, start_y)
			split_img = img[:,start_y:start_y + pic_size,start_x:start_x + pic_size]
			split_imgs.append(split_img)
			if (start_x <= center[0] and center[0] < start_x + pic_size 
						and start_y <= center[1] and center[1] < start_y + pic_size): # Make sure center is in the picture
				split_imgs_labels.append(1)
			else:
				split_imgs_labels.append(0)
			# plt.imshow(split_img.transpose((1,2,0)))
			# plt.show()
	return split_imgs, split_imgs_labels # returns it in batch_size x depth x 10 x 10

def img_partition_collector_regression(img, label, iters=2000):
	"""
	img is assumed to be 3 x 160 x 120, Depth x Height x Width
	label is assumed to be x,y coordinate of robot center
	start_coord is in pixels
	pic_size is in pixels
	"""
	# Calculate the center of the picture relative to pixels
	center = (label[0], label[1]) # This is in real units ==> Convert to pixels
	center = ((center[0] - top_left[0])/scaling_x, (top_left[1] - center[1])/scaling_y)
	split_imgs = []
	split_imgs_labels = []
	for _ in range(iters):
		good_start_location = False
		while(not good_start_location):
			start_coord = (np.random.uniform(center[0] - 10, center[0]), np.random.uniform(center[1] - 10, center[1]))
			start_x = int(start_coord[0])
			start_y = int(start_coord[1])
			split_img = img[:,start_y:start_y + pic_size,start_x:start_x + pic_size]
			if (start_x <= center[0] and center[0] < start_x + pic_size
						and start_y <= center[1] and center[1] < start_y + pic_size): # Make sure center is in the picture
				relative_coord = np.array([(center[0] - start_x) * scaling_x, - (start_y - center[1]) * scaling_y]) # relative to top left
				split_imgs_labels.append(relative_coord)
				split_imgs.append(split_img)
				good_start_location = True
			# else:
			# 	print("Bad start location")
	return split_imgs, split_imgs_labels