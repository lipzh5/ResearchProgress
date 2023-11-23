# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: transform episodic data into trajectories
# refer to: https://colab.research.google.com/drive/1XH82FPp22-ho-HAwvephJvMXoXdp6il_#scrollTo=Dgf1OxIhJwib
# https://robotics-transformer-x.github.io/

from typing import Any, Dict, Union, NamedTuple, Optional
import abc
import dataclasses
import reverb
import tensorflow as tf
from rlds import rlds_types
from rlds import transformations
# from robotics_transformer.train_rt1.trajectory_transform import rlds_spec, utils
from rlds_spec import RLDS_SPEC, TENSOR_SPEC
from utils import create_structured_writer_config
import functools
import tf_agents
import numpy as np



@dataclasses.dataclass
class TrajectoryTransformer(metaclass=abc.ABCMeta):
  # a set of rules transforming a dataset of RLDS episodes to a dataset of trajectories.
  episode_dataset_spec: RLDS_SPEC # TODO
  episode_to_steps_fn_dataset_spec: RLDS_SPEC
  steps_dataset_spec: Any
  pattern: reverb.structured_writer.Pattern
  episode_to_steps_map_fn: Any
  expected_tensor_spec: TENSOR_SPEC
  step_map_fn: Optional[Any] = None
  
  def get_for_cached_trajectory_transform(self):
    """Creates a copy of this traj transform to use with caching.

    The returned TrajectoryTransfrom copy will be initialized with the default
    version of the `episode_to_steps_map_fn`, because the effect of that
    function has already been materialized in the cached copy of the dataset.
    Returns:
      trajectory_transform: A copy of the TrajectoryTransform with overridden
        `episode_to_steps_map_fn`.
    """
    traj_copy = dataclasses.replace(self)
    traj_copy.episode_dataset_spec = traj_copy.episode_to_steps_fn_dataset_spec
    traj_copy.episode_to_steps_map_fn = lambda e: e[rlds_types.STEPS]
    return traj_copy
  
  def transform_episodic_rlds_dataset(self, episodes_dataset: tf.data.Dataset):
    """Applies this TrajectoryTransform to the dataset of episodes."""

    # Convert the dataset of episodes to the dataset of steps.
    steps_dataset = episodes_dataset.map(
        self.episode_to_steps_map_fn, num_parallel_calls=tf.data.AUTOTUNE
    ).flat_map(lambda x: x)
    print('transform episodic rlds dataset!!! ')
    for d in steps_dataset:
      print(type(d), d)
      # print(d['steps'])
      break

    return self._create_pattern_dataset(steps_dataset)
  
  # add by pz
  def transform_episodic_rlds_datasource(self, episodes_datasource):
    # steps_datasource = tf.nest.map_structure(self.episode_to_steps_map_fn, episodes_datasource)
    from training_helper import get_spec_from_file
    policy_specs = get_spec_from_file()

    def step_generator():
      for ep in episodes_datasource:
        for step in ep['steps']:
          yield step
          # yield {k: v for k, v in step.items() if k!='info'}

    # action', 'info', 'is_first', 'is_last', 'is_terminal', 'num_steps', 'observation', 'reward', 'step_id'
    output_types = {'action': policy_specs['action_spec'], 
                    'observation': policy_specs['time_step_spec'].observation,
                    'info': policy_specs['info_spec'],
                    'is_first': tf_agents.specs.ArraySpec(shape=(), dtype=bool, name='is_first'),
                    'is_last': tf_agents.specs.ArraySpec(shape=(), dtype=bool, name='is_last'),
                    'is_terminal': tf_agents.specs.ArraySpec(shape=(), dtype=bool, name='is_terminal'),
                    'num_steps': tf_agents.specs.ArraySpec(shape=(), dtype=np.int32, name='num_steps'),
                    'reward': policy_specs['time_step_spec'].reward,
                    'step_id': tf_agents.specs.ArraySpec(shape=(), dtype=np.int32, name='step_id')}

    # steps_dataset = tf.data.Dataset.from_generator(functools.partial(step_generator), output_types=output_types)
    for k, v in output_types.items():
      output_types[k] = tf.nest.map_structure(tf.TensorSpec.from_spec,
      tf.nest.map_structure(tf_agents.specs.tensor_spec.from_spec, v))
    steps_dataset = tf.data.Dataset.from_generator(
      functools.partial(step_generator),
      output_signature=output_types
    )   
    return self._create_pattern_dataset(steps_dataset)
 
  
  def transform_steps_rlds_dataset(
      self, steps_dataset: tf.data.Dataset
  ) -> tf.data.Dataset:
    """Applies this TrajectoryTransform to the dataset of episode steps."""

    return self._create_pattern_dataset(steps_dataset)

  def create_test_dataset(
      self,
  ) -> tf.data.Dataset:
    """Creates a test dataset of trajectories.

    It is guaranteed that the structure of this dataset will be the same as
    when flowing real data. Hence this is a useful construct for tests or
    initialization of JAX models.
    Returns:
      dataset: A test dataset made of zeros structurally identical to the
        target dataset of trajectories.
    """
    zeros = transformations.zeros_from_spec(self.expected_tensor_spec)

    return tf.data.Dataset.from_tensors(zeros)

  def _create_pattern_dataset(self, steps_dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Create PatternDataset from the `steps_dataset`."""
    config = create_structured_writer_config('temp', self.pattern)

    # Further transform each step if the `step_map_fn` is provided.
    if self.step_map_fn:
      steps_dataset = steps_dataset.map(self.step_map_fn)
    
    pattern_dataset = reverb.PatternDataset(
        input_dataset=steps_dataset,
        configs=[config],
        respect_episode_boundaries=True,
        is_end_of_episode=lambda x: x[rlds_types.IS_LAST])
    return pattern_dataset









