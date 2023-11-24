# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import numpy as np
from PickPlaceEnv import PickPlaceEnv
import matplotlib.pyplot as plt
import ViLD


config = {'pick': ['yellow block', 'green clock', 'blue block'],
		  'place': ['yellow bowl', 'green bowl', 'blue bowl']}


def show_env_init():
	np.random.seed(42)
	env = PickPlaceEnv()
	print('instantiate env!!!')
	obs = env.reset(config)
	plt.subplot(1, 2, 1)
	img = env.get_camera_image()
	print('env.get image: ', img)
	plt.title('Perspective side-view')
	plt.imshow(img)
	plt.subplot(1, 2, 2)
	img = env.get_camera_image_top()
	img = np.flipud(img.transpose(1, 0, 2))
	plt.title('Orthographic top-view')
	plt.imshow(img)
	plt.show()
	# note: orthographic cameras do not exist. but we can approximate them by projecting
	# a 3D point could from an RGB-D camera, then unprojecting that onto
	# an orthographic plane. Orthographic views are useful for spatial action maps.
	plt.title('Unprojected orthographic top-view')
	plt.imshow(obs['image'])
	plt.show()
	pass



if __name__ == "__main__":
	import pybullet
	# download_clip()
	show_env_init()

