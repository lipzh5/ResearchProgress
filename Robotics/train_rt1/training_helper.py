# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import os
import tensorflow as tf
import numpy as np
from tf_agents.policies import policy_saver
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import Trajectory
from tensor2robot.utils import tensorspec_utils
import transformer_network
import sequence_agent
from trajectory_transformer_builder import TrajectoryTransformBuilder
from rlds_spec import RLDS_SPEC, TENSOR_SPEC
from utils import n_step_pattern_builder, step_map_fn
import importlib



# from tf_agents.trajectories import StepType
# from tf_agents.trajectories import trajectory

BATCH_SIZE = 2
TIME_SEQUENCE_LENGTH = 6


# get spec from file
def get_spec_from_file(rel_path='robotics_transformer/trained_checkpoints/rt1main'):
  print('current work dir: ', os.getcwd())
  model_path = os.path.join(os.getcwd(), rel_path)
  spec_path = os.path.join(model_path, policy_saver.POLICY_SPECS_PBTXT)

  policy_specs = policy_saver.specs_from_collect_data_spec(
  tensor_spec.from_pbtxt_file(spec_path))

  loaded_time_step_spec = policy_specs['time_step_spec'] # loaded from file
  loaded_action_spec = policy_specs['action_spec']
  policy_state_spec = policy_specs['policy_state_spec']
  info_spec = policy_specs['info_spec']
  # print(policy_specs.keys())

  return policy_specs


def get_action_state_type_spec(rel_path='robotics_transformer/trained_checkpoints/rt1main'):
  policy_specs = get_spec_from_file(rel_path)
  # state type spec
  observation = policy_specs['time_step_spec'].observation
  state_type_spec = tensorspec_utils.TensorSpecStruct()
  for k, v in observation.items():
    setattr(state_type_spec, k, tensor_spec.from_spec(v))
  # action type spec
  action_type_spec = tensorspec_utils.TensorSpecStruct()
  for k, v in policy_specs['action_spec'].items():
    setattr(action_type_spec, k, tensor_spec.from_spec(v))
  return action_type_spec, state_type_spec


def create_agent(actor_network=None, rel_path='robotics_transformer/trained_checkpoints/rt1main'):
  action_type_spec, state_type_spec = get_action_state_type_spec(rel_path)
  if actor_network is None:
    actor_network = transformer_network.TransformerNetwork
  time_step_spec = ts.time_step_spec(observation_spec=state_type_spec)

  agent = sequence_agent.SequenceAgent(
      time_step_spec=time_step_spec,
      action_spec=action_type_spec,
      actor_network=actor_network,
      actor_optimizer=tf.keras.optimizers.Adam(),
      train_step_counter=tf.compat.v1.train.get_or_create_global_step(),
      time_sequence_length=TIME_SEQUENCE_LENGTH,
  )
  agent.initialize()
  return agent

# ====== just for fun ⬇ =====
def get_experience(data_source, ep_id=0):
  eps = data_source[ep_id] # {'steps': [...]}

  steps = eps['steps']
  image = steps[0]['observation']['image']
  src_rot = steps[0]['observation']['src_rotation']
  action = steps[0]['action'] 
  observation = steps[0]['observation']
  info = steps[0]['info']
  reward = steps[0]['reward']
  nested_action, nested_obs, nested_info, nested_reward = get_nested_struct(
    action, observation, info, reward)

  nested_step_type = tf.constant(1, dtype=tf.int32)
  nested_step_type = tf.nest.map_structure(lambda t: tf.stack([t]*TIME_SEQUENCE_LENGTH), nested_step_type)
  nested_step_type = tf.nest.map_structure(lambda t: tf.stack([t]*BATCH_SIZE), nested_step_type)

  nested_reward = tf.constant(0., dtype=tf.float32)
  nested_reward = tf.nest.map_structure(lambda t: tf.stack([t]*TIME_SEQUENCE_LENGTH), nested_reward)
  nested_reward = tf.nest.map_structure(lambda t: tf.stack([t]*BATCH_SIZE), nested_reward)

  nested_discount = tf.constant(0., dtype=tf.float32)
  nested_discount = tf.nest.map_structure(lambda t: tf.stack([t]*TIME_SEQUENCE_LENGTH), nested_discount)
  nested_discount = tf.nest.map_structure(lambda t: tf.stack([t]*BATCH_SIZE), nested_discount)
  # TODO test
  return Trajectory(
    step_type=nested_step_type,
    next_step_type=nested_step_type,
    reward=nested_reward, #tf.constant(1., dtype=tf.float32),
    policy_info=nested_info,
    observation=nested_obs,
    action=nested_action,
    discount=nested_discount, #tf.constant(0., tf.float32),
  )

  

def get_nested_struct(action, observation, info, reward):
  nest = tf.nest.map_structure(lambda t: tf.stack([t]*TIME_SEQUENCE_LENGTH), action)
  nested_action = tf.nest.map_structure(lambda t: tf.stack([t]*BATCH_SIZE), nest)

  # observation['image'] = observation['image'].astype(np.uint8)
  nest = tf.nest.map_structure(lambda t: tf.stack([t]*TIME_SEQUENCE_LENGTH), observation)
  nested_obs = tf.nest.map_structure(lambda t: tf.stack([t]*BATCH_SIZE), nest)
  nested_obs['image'] = tf.cast(nested_obs['image'], tf.uint8)
  # print('observation image type: ', observation['image'].dtype)
  

  nest = tf.nest.map_structure(lambda t: tf.stack([t]*TIME_SEQUENCE_LENGTH), info)
  nested_info = tf.nest.map_structure(lambda t: tf.stack([t]*BATCH_SIZE), nest)

  nest = tf.nest.map_structure(lambda t: tf.stack([t]*TIME_SEQUENCE_LENGTH), reward)
  nested_reward = tf.nest.map_structure(lambda t: tf.stack([t]*BATCH_SIZE), nest)
  return nested_action, nested_obs, nested_info, nested_reward

# ====== just for fun ⬆ =====

def get_trajectory_dataset(data_source, step_features):
  
  rt1_rlds_spec = RLDS_SPEC(
    observation_info=step_features['observation'],
    action_info=step_features['action'],
    reward_info=step_features['reward'],
    step_metadata_info=step_features['info'])
  traj_transformer = TrajectoryTransformBuilder(rt1_rlds_spec, step_map_fn=step_map_fn, pattern_fn=n_step_pattern_builder(TIME_SEQUENCE_LENGTH)).build(validate_expected_tensor_spec=False)
  traj_dataset = traj_transformer.transform_episodic_rlds_datasource(data_source)
  return traj_dataset
  
  


  
  




