# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
# Gripper (robotic 2F85)
import numpy as np
import pybullet
import threading
import time
import os


class Robotiq2F85:

	def __init__(self, robot, tool):
		self.robot = robot
		self.tool = tool
		pos = [0.1339999999999999, -0.49199999999872496, 0.5]
		rot = pybullet.getQuaternionFromEuler([np.pi, 0, np.pi])
		#urdf = os.path.join(os.getcwd(), '../../robotiq_2f_85/robotiq_2f_85.urdf')
		# urdf = 'robotiq_2f_85/robotiq_2f_85.urdf'
		urdf = '/mnt/sdd/PycharmProjects/RoboticsSimulation/robotiq_2f_85/robotiq_2f_85.urdf'
		self.body = pybullet.loadURDF(urdf, pos, rot)
		self.n_joints = pybullet.getNumJoints(self.body)
		self.activated = False

		# connect gripper base to robot tool
		pybullet.createConstraint(self.robot, tool, self.body, 0, jointType=pybullet.JOINT_FIXED, jointAxis=[0, 0, 0],
								  parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, -0.07],
								  childFrameOrientation=pybullet.getQuaternionFromEuler([0, 0, np.pi / 2]))
		# set friction coefficients for grippers fingers
		for i in range(pybullet.getNumJoints(self.body)):
			pybullet.changeDynamics(self.body, i, lateralFriction=10.0, spinningFriction=1.0,
									rollingFriction=1.0, frictionAnchor=True)

		# start thread to handle additional gripper constraints.
		self.motor_joint = 1
		self.constrains_thread = threading.Thread(target=self.step)
		self.constrains_thread.daemon = True
		self.constrains_thread.start()

	# control joint positions by enforcing hard constraints on gripper behavior
	# set one joint as the open/close motor joint (other joints should mimic)
	def step(self):
		while True:
			try:
				currj = [pybullet.getJointState(self.body, i)[0] for i in range(self.n_joints)]
				indj = [6, 3, 8, 5, 10]
				targj = [currj[1], -currj[1], -currj[1], currj[1], currj[1]]
				pybullet.setJointMotorControlArray(self.body, indj, pybullet.POSITION_CONTROL, targj,
												   positionGains=np.ones(5))
			except:
				return
			time.sleep(0.001)

	# close gripper fingers
	def activate(self):
		pybullet.setJointMotorControl2(self.body, self.motor_joint, pybullet.VELOCITY_CONTROL, targetVelocity=1,
									   force=10)
		self.activated = True

	# open gripper fingers
	def release(self):
		pybullet.setJointMotorControl2(self.body, self.motor_joint, pybullet.VELOCITY_CONTROL, targetVelocity=-1,
									   force=10)
		self.activated = False

	# if activated and object in gripper: check object contact
	# if activated and nothing in gripper: check gripper contact
	# If released: check proximity to surface (disabled)
	def detect_contact(self):
		obj, _, ray_frac = self.check_proximity()
		if self.activated:
			empty = self.grasp_width() < 0.01
			cbody = self.body if empty else obj
			if obj == self.body or obj == 0:
				return False
			return self.external_contact(cbody)

	# return if body is in contact with something other than gripper
	def external_contact(self, body=None):
		if body is None:
			body = self.body
		pts = pybullet.getContactPoints(bodyA=body)
		pts = [pt for pt in pts if pt[2] != self.body]
		return len(pts) > 0

	def check_grasp(self):
		while self.moving():
			time.sleep(0.001)
		success = self.grasp_width() > 0.01
		return success

	def grasp_width(self):
		lpad = np.array(pybullet.getLinkState(self.body, 4)[0])
		rpad = np.array(pybullet.getLinkState(self.body, 9)[0])
		dist = np.linalg.norm(lpad - rpad) - 0.047813
		return dist

	def check_proximity(self):
		ee_pos = np.array(pybullet.getLinkState(self.robot, self.tool)[0])
		tool_pos = np.array(pybullet.getLinkState(self.body, 0)[0])
		vec = (tool_pos - ee_pos) / np.linalg.norm((tool_pos - ee_pos))  # direction
		ee_targ = ee_pos + vec
		ray_data = pybullet.rayTest(ee_pos, ee_targ)[0]
		obj, link, ray_frac = ray_data[0], ray_data[1], ray_data[2]
		return obj, link, ray_frac
