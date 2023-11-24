# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None

import os
import numpy as np
from urllib.parse import ParseResultBytes
import pybullet
import pybullet_data
from Robotiq2F85 import Robotiq2F85
from const import BOUNDS, COLORS, PIXEL_SIZE


class PickPlaceEnv():

	def __init__(self):
		self.dt = 1 / 480
		self.sim_step = 0

		# Configure and start PyBullet
		# python3 -m pybullet_utils.runServer
		# pybullet.connect(pybullet.SHARED_MEMORY) # pybullet.GUI for local GUI
		pybullet.connect(pybullet.DIRECT)  # pybullet.GUI for local GUI
		pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
		pybullet.setPhysicsEngineParameter(enableFileCaching=0)
		assets_path = os.path.dirname(os.path.abspath(''))
		pybullet.setAdditionalSearchPath(assets_path)
		pybullet.setTimeStep(self.dt)

		# Joint angles: (J0, J1, J2, J3, J4, J5)
		self.home_joints = (np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, 3 * np.pi / 2, 0)
		self.home_ee_euler = (np.pi, 0, np.pi)  # (RX, RY, RZ) rotation in Euler angles
		self.ee_link_id = 9  # Link ID of UR5 end effector
		self.tip_link_id = 10  # Link ID of gripper finger tips
		self.gripper = None

	def reset(self, config):
		pybullet.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)
		pybullet.setGravity(0, 0, -9.8)
		self.cache_video = []

		# Temporarily disable rendering to load URDFs faster
		pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

		# Add robot
		pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
		pybullet.loadURDF('plane.urdf', [0, 0, -0.001])
		import os
		print('urdf path found: ', os.path.exists('ur5e/ur5e.urdf'))
		print('current work dir: ', os.getcwd())
		# current_cwd = os.getcwd()
		# urdf_file_path = os.path.join(os.getcwd(), '../../ur5e/ur5e.urdf')
		urdf_file_path = '/mnt/sdd/PycharmProjects/RoboticsSimulation/ur5e/ur5e.urdf'
		print('urdf filepath exits? ', os.path.exists(urdf_file_path))
		self.robot_id = pybullet.loadURDF(urdf_file_path, [0, 0, 0], flags=pybullet.URDF_USE_MATERIAL_COLORS_FROM_MTL)
		self.ghost_id = pybullet.loadURDF(urdf_file_path, [0, 0, -10])  # for forward kinematics
		self.joint_ids = [pybullet.getJointInfo(self.robot_id, i) for i in range(pybullet.getNumJoints(self.robot_id))]
		self.joint_ids = [j[0] for j in self.joint_ids if j[2] == pybullet.JOINT_REVOLUTE]

		# Move robot to home configuration
		# for i in range(len(self.joint_ids))
		for i, jid in enumerate(self.joint_ids):
			pybullet.resetJointState(self.robot_id, jid, self.home_joints[i])
			pass
		# Add gripper
		if self.gripper is not None:
			while self.gripper.constrains_thread.is_alive():
				self.constrains_thread_active = False
		self.gripper = Robotiq2F85(self.robot_id, self.ee_link_id)
		self.gripper.release()

		# Add workspace
		plane_shape = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[0.3, 0.3, 0.001])
		plane_visual = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[0.3, 0.3, 0.001])
		plane_id = pybullet.createMultiBody(0, plane_shape, plane_visual, basePosition=[0, -0.5, 0])
		pybullet.changeVisualShape(plane_id, -1, rgbaColor=[0.2, 0.2, 0.2, 1.0])

		# load objects according to config
		self.config = config
		self.obj_name_to_id = {}
		obj_names = list(self.config['pick']) + list(self.config['place'])
		obj_xyz = np.zeros((0, 3))
		for obj_name in obj_names:
			if ('block' in obj_name) or ('bowl' in obj_name):
				# get random position 15cm+ from other objects
				while True:
					rand_x = np.random.uniform(BOUNDS[0, 0] + 0.1, BOUNDS[0, 1] - 0.1)
					rand_y = np.random.uniform(BOUNDS[1, 0] + 0.1, BOUNDS[1, 1] - 0.1)
					rand_xyz = np.float32([rand_x, rand_y, 0.03]).reshape(1, 3)
					if len(obj_xyz) == 0:
						obj_xyz = np.concatenate((obj_xyz, rand_xyz), axis=0)
						break
					else:
						nn_dist = np.min(np.linalg.norm(obj_xyz - rand_xyz, axis=1)).squeeze()
						if nn_dist > 0.15:  # ensure 15cm+ away from others
							obj_xyz = np.concatenate((obj_xyz, rand_xyz), axis=0)
							break

				object_color = COLORS[obj_name.split(' ')[0]]
				object_type = obj_name.split(' ')[1]
				object_position = rand_xyz.squeeze()
				if object_type == 'block':
					object_shape = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
					object_visual = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
					object_id = pybullet.createMultiBody(0.01, object_shape, object_visual,
														 basePosition=object_position)
				else:
					object_position[2] = 0
					#urdf_file_path = os.path.join(os.getcwd(), '../../bowl/bowl.urdf')
					urdf_file_path = '/mnt/sdd/PycharmProjects/RoboticsSimulation/bowl/bowl.urdf'
					object_id = pybullet.loadURDF(urdf_file_path, object_position, useFixedBase=1)
				pybullet.changeVisualShape(object_id, -1, rgbaColor=object_color)
				self.obj_name_to_id[obj_name] = object_id

		# re-enable rendering
		pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

		for _ in range(200):
			pybullet.stepSimulation()
		return self.get_observation()

	def servoj(self, joints):
		# move to target joint positions with position control
		pybullet.setJointMotorControlArray(
			bodyIndex=self.robot_id,
			jointIndices=self.joint_ids,
			controlMode=pybullet.POSITION_CONTROL,
			targetPositions=joints,
			positionGains=[0.01] * 6
		)

	def movep(self, position):
		# move to target end effector position
		joints = pybullet.calculateInverseKinematics(
			bodyUniqueId=self.robot_id,
			endEffectorLinkIndex=self.tip_link_id,
			targetPosition=position,
			targetOrientation=pybullet.getQuaternionFromEuler(self.home_ee_euler),
			maxNumIterations=100)
		self.servoj(joints)

	def step(self, action=None):
		# do pick and place motion primitive
		pick_xyz, place_xyz = action['pick'].copy(), action['place'].copy()

		# set fixed primitive z-heights
		hover_xyz = pick_xyz.copy() + np.float32([0, 0, 0.2])
		pick_xyz[2] = 0.03
		place_xyz[2] = 0.15

		# move to object
		ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
		while np.linalg.norm(hover_xyz - ee_xyz) > 0.01:
			self.movep(hover_xyz)
			self.step_sim_and_render()
			ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
		while np.linalg.norm(pick_xyz - ee_xyz) > 0.01:
			self.movep(pick_xyz)
			self.step_sim_and_render()
			ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])

		# pick up object
		self.gripper.activate()
		for _ in range(240):
			self.step_sim_and_render()
		while np.linalg.norm(hover_xyz - ee_xyz) > 0.01:
			self.movep(hover_xyz)
			self.step_sim_and_render()
			ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])

		# move to place location
		while np.linalg.norm(place_xyz - ee_xyz) > 0.01:
			self.movep(place_xyz)
			self.step_sim_and_render()
			ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])

		# place down object
		while (not self.gripper.detect_contact()) and (place_xyz[2]) > 0.03:
			place_xyz[2] -= 0.001
			self.movep(place_xyz)
			for _ in range(3):
				self.step_sim_and_render()
		self.gripper.release()
		for _ in range(240):
			self.step_sim_and_render()
		place_xyz[2] = 0.2
		ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
		# move up
		while np.linalg.norm(place_xyz - ee_xyz) > 0.01:
			self.movep(place_xyz)
			self.step_sim_and_render()
			ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
		# move to center
		place_xyz = np.float32([0, -0.5, 0.2])
		while np.linalg.norm(place_xyz - ee_xyz) > 0.01:
			self.movep(place_xyz)
			self.step_sim_and_render()
			ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])

		observation = self.get_observation()
		reward = self.get_reward()
		done = False
		info = {}
		print('env step success!!!!!')
		print('###'*4)
		return observation, reward, done, info

	def set_alpha_transparency(self, alpha):
		for _id in range(20):
			visual_shape_data = pybullet.getVisualShapeData(_id)
			for i in range(len(visual_shape_data)):
				object_id, link_index, _, _, _, _, _, rgba_color = visual_shape_data[i]
				rgba_color = list(rgba_color[0:3]) + [alpha]
				pybullet.changeVisualShape(
					self.robot_id, linkIndex=i, rgbaColor=rgba_color)
				pybullet.changeVisualShape(self.gripper.body, linkIndex=i, rgbaColor=rgba_color)

	def step_sim_and_render(self):
		pybullet.stepSimulation()
		self.sim_step += 1

		# render current image at 8 FPS
		if self.sim_step % 60 == 0:
			self.cache_video.append(self.get_camera_image())

	def get_camera_image(self):
		image_size = (240, 240)
		intrinsics = (120., 0, 120., 0, 120., 120., 0, 0, 1)
		# modified by pz
		color, _, _, _, _ = self.render_image(image_size, intrinsics)
		# color, _, _, _, _ = env.render_image(image_size, intrinsics)
		return color

	def get_camera_image_top(self,
							 image_size=(240, 240),
							 intrinsics=(2000., 0, 2000., 0, 2000., 2000., 0, 0, 1),
							 position=(0, -0.5, 5),
							 orientation=(0, np.pi, -np.pi / 2),
							 zrange=(0.01, 1.),
							 set_alpha=True):
		set_alpha and self.set_alpha_transparency(0)
		# modified by pz
		color, _, _, _, _ = self.render_image_top(image_size, intrinsics, position, orientation, zrange)
		# color, _, _, _, _ = env.render_image_top(image_size, intrinsics, position, orientation, zrange)
		set_alpha and self.set_alpha_transparency(1)
		return color

	def get_reward(self):
		return 0  # TODO: check did the robot follow text instructions?

	def get_observation(self):
		observation = {}
		# render current image
		color, depth, position, orientation, intrinsics = self.render_image()
		# get heightmaps and colormaps
		points = self.get_pointcloud(depth, intrinsics)
		position = np.float32(position).reshape(3, 1)
		rotation = pybullet.getMatrixFromQuaternion(orientation)
		rotation = np.float32(rotation).reshape(3, 3)
		transform = np.eye(4) # diagonal matrix
		transform[:3, :] = np.hstack((rotation, position))
		points = self.transform_pointcloud(points, transform)
		heightmap, colormap, xyzmap = self.get_heightmap(points, color, BOUNDS, PIXEL_SIZE)

		observation['image'] = colormap
		observation['xyzmap'] = xyzmap
		observation['pick'] = list(self.config['pick'])
		observation['place'] = list(self.config['place'])
		return observation

	def render_image(self, image_size=(720, 720), intrinsics=(360.,0, 360., 0, 360., 360., 0, 0, 1)):
		# camera parameters
		position = (0, -0.85, 0.4)
		orientation = (np.pi / 4 + np.pi / 48, np.pi, np.pi)
		orientation = pybullet.getQuaternionFromEuler(orientation)
		zrange = (0.01, 10.)
		noise = True

		# OpenGL camera settings
		lookdir = np.float32([0, 0, 1]).reshape(3, 1)
		updir = np.float32([0, -1, 0]).reshape(3, 1)
		rotation = pybullet.getMatrixFromQuaternion(orientation)
		rotm = np.float32(rotation).reshape(3, 3)
		lookdir = (rotm @ lookdir).reshape(-1)
		updir = (rotm @ updir).reshape(-1)
		lookat = position + lookdir
		focal_len = intrinsics[0]
		znear, zfar = (0.01, 10.)
		viewm = pybullet.computeViewMatrix(position, lookat, updir)
		fovh = (image_size[0] / 2) / focal_len
		fovh = 180 * np.arctan(fovh) * 2 / np.pi
		# Notes: 1) FOV is vertical FOV 2) aspect must be float
		aspect_ratio = image_size[1] / image_size[0]
		projm = pybullet.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

		# render with OpenGL camera settings
		_, _, color, depth, segm = pybullet.getCameraImage(width=image_size[1],
														   height=image_size[0],
														   viewMatrix=viewm,
														   projectionMatrix=projm,
														   shadow=1,
														   flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
														   renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
		# Get color image
		color_image_size = (image_size[0], image_size[1], 4)
		color = np.array(color, dtype=np.uint8).reshape(color_image_size)
		color = color[:, :, :3] # remove alpha channel
		if noise:
			color = np.int32(color)
			color += np.int32(np.random.normal(0, 3, color.shape))
			color = np.uint8(np.clip(color, 0, 255))
		# get depth image
		depth_image_size = (image_size[0], image_size[1])
		zbuffer = np.float32(depth).reshape(depth_image_size)
		depth = (zfar + znear - (2 * zbuffer - 1) * (zfar - znear))
		depth = (2 * znear * zfar) / depth
		if noise:
			depth += np.random.normal(0, 0.003, depth.shape)
		intrinsics = np.float32(intrinsics).reshape(3, 3)
		return color, depth, position, orientation, intrinsics

	def render_image_top(self, image_size=(240, 240),
						 intrinsics=(2000., 0, 2000., 0, 2000., 2000., 0, 0, 1),
						 position=(0, -0.5, 5),
						 orientation=(0, np.pi, -np.pi/2),
						 zrange=(0.01, 1.)):
		# camera parameters
		orientation = pybullet.getQuaternionFromEuler(orientation)
		noise = True
		# OpenGL camera settings
		lookdir = np.float32([0, 0, 1]).reshape(3, 1)
		updir = np.float32([0, -1, 0]).reshape(3, 1)
		rotation = pybullet.getMatrixFromQuaternion(orientation)
		rotm = np.float32(rotation).reshape(3, 3)
		lookdir = (rotm @ lookdir).reshape(-1)
		updir = (rotm @ updir).reshape(-1)
		lookat = position + lookdir
		focal_len = intrinsics[0]
		znear, zfar = (0.01, 10.)
		viewm = pybullet.computeViewMatrix(position, lookat, updir)
		fovh = (image_size[0] / 2) / focal_len
		fovh = 180 * np.arctan(fovh) * 2 / np.pi
		# note: 1) FOV is vertival FOV 2) aspect must be float
		aspect_ratio = image_size[1] / image_size[0]
		projm = pybullet.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)
		# render with OpenGL camera settings
		_, _, color, depth, segm = pybullet.getCameraImage(width=image_size[1],
														   height=image_size[0],
														   viewMatrix=viewm,
														   projectionMatrix=projm,
														   shadow=1,
														   flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
														   renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
		# get color image
		color_image_size = (image_size[0], image_size[1], 4)
		color = np.array(color, dtype=np.uint8).reshape(color_image_size)
		color = color[:, :, :3] # remove alpha channel
		if noise:
			color = np.int32(color)
			color += np.int32(np.random.normal(0, 3, color.shape))
			color = np.uint8(np.clip(color, 0, 255))
		# get depth image
		depth_image_size = (image_size[0], image_size[1])
		zbuffer = np.float32(depth).reshape(depth_image_size)
		depth = (zfar + znear - (2 * zbuffer -1) * (zfar - znear))
		depth = (2 * znear * zfar) / depth
		if noise:
			depth += np.random.normal(0, 0.003, depth.shape)
		intrinsics = np.float32(intrinsics).reshape(3, 3)
		return color, depth, position, orientation, intrinsics

	def get_pointcloud(self, depth, intrinsics):
		# get 3D pointcloud from perspective depth image
		# args:
		#  depth: HxW float array of perspective depth in meters
		#  intrinsics: 3x3 float array of camera intrinsics matrix.
		# returns:
		#  points: HxWx3 float array of 3D points in camera coordinates
		height, width = depth.shape
		xlin = np.linspace(0, width - 1, width)
		ylin = np.linspace(0, height - 1, height)
		px, py = np.meshgrid(xlin, ylin)
		px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
		py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
		points = np.float32([px, py, depth]).transpose(1, 2, 0)
		return points

	def transform_pointcloud(self, points, transform):
		'''
		Apply rigid transformation to 3D pointcloud
		:param points: HxWx3 float array of 3D points in camera coordinates
		:param transform: 4x4 float array representing a rigid transformation matrix.
		:return: HxWx3 float array of transformed 3D points
		'''
		padding = ((0, 0), (0, 0), (0, 1))
		homogen_points = np.pad(points.copy(), padding,
								'constant', constant_values=1)
		for i in range(3):
			points[Ellipsis, i] = np.sum(transform[i, :] * homogen_points, axis=-1)
		return points

	def get_heightmap(self, points, colors, bounds, pixel_size):
		'''
		get top-down (z-axis) orthographic heightmap image from 3D pointcloud.
		:param points: HxWx3 float array of 3D points in world coordinates
		:param colors: HxWx3 uint8 array of values in range 0-255 aligned with points
		:param bounds: 3x2 float array of values (row: X,Y,Z; columns: min, max) defining
		region in 3D space to generate heightmap in world coordinates.
		:param pixel_size: float defining size of each pixel in meters
		:return: heightmap: HxW float array of height (from lower z-bound) in meters
		colormap: HxWx3 uint8 array of backprojected color aligned with heightmap
		xyzmap: HxWx3 float array of XYZ points in world coordinates
		'''
		width = int(np.round((bounds[0, 1]-bounds[0, 0]) / pixel_size))
		height = int(np.round((bounds[1, 1]-bounds[1, 0]) / pixel_size))
		heightmap = np.zeros((height, width), dtype=np.float32)
		colormap = np.zeros((height, width, colors.shape[-1]), dtype=np.uint8)
		xyzmap = np.zeros((height, width, 3), dtype=np.float32)
		# filter out 3D points that are outside of the predefined bounds
		ix = (points[Ellipsis, 0] >= bounds[0, 0]) & (points[Ellipsis, 0] < bounds[0, 1])
		iy = (points[Ellipsis, 1] >= bounds[1, 0]) & (points[Ellipsis, 1] < bounds[1, 1])
		iz = (points[Ellipsis, 2] >= bounds[2, 0]) & (points[Ellipsis, 2] < bounds[2, 1])
		valid = ix & iy & iz
		points = points[valid]
		colors = colors[valid]
		# sort 3D points by z-value, which works with array assignment to simulate
		# z-buffering for rendering the heightmap image
		iz = np.argsort(points[:, -1])
		points, color = points[iz], colors[iz]
		px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size))
		py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size))
		px = np.clip(px, 0, width - 1)
		py = np.clip(py, 0, height - 1)
		heightmap[py, px] = points[:, 2] - bounds[2, 0]
		for c in range(colors.shape[-1]):
			colormap[py, px, c] = colors[:, c]
			xyzmap[py, px, c] = points[:, c]
		colormap = colormap[::-1, :, :] # Flip up-down
		xv, yv = np.meshgrid(np.linspace(BOUNDS[0, 0], BOUNDS[0, 1], height),
							 np.linspace(BOUNDS[1, 0], BOUNDS[1, 1], width))
		xyzmap[:, :, 0] = xv
		xyzmap[:, :, 1] = yv
		xyzmap = xyzmap[::-1, :, :] # Flip up-down
		heightmap = heightmap[::-1, :] # Flip up-down
		return heightmap, colormap, xyzmap


def download_src():
	import gdown
	if not os.path.exists('ur5e/ur5e.urdf'):
		url_prefix = 'https://drive.google.com/uc?id='
		gdown.download(url=url_prefix+'1Cc_fDSBL6QiDvNT4dpfAEbhbALSVoWcc', output='ur5e.zip')
		gdown.download(url=url_prefix + '1yOMEm-Zp_DL3nItG9RozPeJAmeOldekX', output='robotiq_2f_85.zip')
		gdown.download(url=url_prefix + '1GsqNLhEl9dd4Mc3BM0dX3MibOI1FVWNM', output='bowl.zip')


env = PickPlaceEnv()

if __name__ == '__main__':
	download_src()

	pass



















