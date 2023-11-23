# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: transform episodic data into trajectories
# refer to: https://colab.research.google.com/drive/1XH82FPp22-ho-HAwvephJvMXoXdp6il_#scrollTo=Dgf1OxIhJwib
# https://robotics-transformer-x.github.io/
from typing import Any
import reverb
import tree
from collections import OrderedDict
from tf_agents.trajectories.time_step import StepType



def create_reverb_table_signature(table_name: str, steps_dataset_spec,
                                  pattern: reverb.structured_writer.Pattern) -> reverb.reverb_types.SpecNest:
  config = create_structured_writer_config(table_name, pattern)
  reverb_table_spec = reverb.structured_writer.infer_signature(
      [config], steps_dataset_spec)
  return reverb_table_spec


def create_structured_writer_config(table_name: str,
                                    pattern: reverb.structured_writer.Pattern) -> Any:
  config = reverb.structured_writer.create_config(
      pattern=pattern, table=table_name, conditions=[])
  return config

def n_step_pattern_builder(n: int) -> Any:
  """Creates trajectory of length `n` from all fields of a `ref_step`."""

  def transform_fn(ref_step):
    traj = {}
    for key in ref_step:
      if isinstance(ref_step[key], dict):
        transformed_entry = tree.map_structure(lambda ref_node: ref_node[-n:],
                                               ref_step[key])
        traj[key] = transformed_entry
      else:
        traj[key] = ref_step[key][-n:]

    return traj

  return transform_fn


def step_map_fn(step):  # It works well!!!!
  # print('step map fn step keys: \n', step.keys())
  transformed_step = OrderedDict()
  transformed_step['observation'] = step['observation']
  transformed_step['action'] = step['action']
  transformed_step['reward'] = step['reward']
  # one for trajectory_transformer_builder and another for trajectory_transformer
  try:
    info = step['info']
    transformed_step['return'] = info['return']
    transformed_step['discounted_return'] = info['discounted_return']
  except:
    transformed_step['return'] = step['return']
    transformed_step['discounted_return'] = step['discounted_return']
 
  transformed_step['is_first'] = step['is_first']
  transformed_step['is_last'] = step['is_last']
  transformed_step['is_terminal'] = step['is_terminal']

  transformed_step['num_steps'] = step['num_steps']
  transformed_step['step_id'] = step['step_id']
  # for create trajectory-like input
  step_type = StepType.LAST if step['is_terminal']==True else StepType.MID
  if step['is_first'] == True:
    step_type = StepType.FIRST
  transformed_step['step_type'] = step_type

  # Tensor("args_10:0", shape=(), dtype=int32)
  # "try" because this step_map_fn may be used by a SimbolicTensfor in transformer_builder  
  # try: # noqa 
  #   margin = step['num_steps'].numpy() - step['step_id'].numpy()
  #   next_step_type = StepType.LAST if margin<=3 else StepType.MID  
  #   transformed_step['next_step_type'] = next_step_type
  # except:
  #   print(step['num_steps'])
  #   pass
    
  return transformed_step


