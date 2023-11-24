# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import os
import numpy as np
import pickle
from ScriptedPolicy import ScriptedPolicy
from PickPlaceEnv import PickPlaceEnv
from const import PLACE_TARGETS, PICK_TARGETS
import matplotlib.pyplot as plt

def collect_demonst():
  env = PickPlaceEnv()
  #@markdown Collect demonstrations with a scripted expert, or download a pre-generated dataset.
  load_pregenerated = True  #@param {type:"boolean"}

  # Load pre-existing dataset.
  if load_pregenerated:
    if not os.path.exists('dataset-9999.pkl'):
      import gdown
      url_prefix = 'https://drive.google.com/uc?id='
      gdown.download(url=url_prefix + '1yCz6C-6eLWb4SFYKdkM-wz5tlMjbG2h8')
      # !gdown --id 1TECwTIfawxkRYbzlAey0z1mqXKcyfPc-
      # !gdown --id 1yCz6C-6eLWb4SFYKdkM-wz5tlMjbG2h8
    dataset = pickle.load(open('dataset-9999.pkl', 'rb'))  # ~10K samples.
    dataset_size = len(dataset['text'])
  return dataset, dataset_size

  # Generate new dataset.
  # else:
  #   dataset = {}
  #   dataset_size = 2  # Size of new dataset.
  #   dataset['image'] = np.zeros((dataset_size, 224, 224, 3), dtype=np.uint8)
  #   dataset['pick_yx'] = np.zeros((dataset_size, 2), dtype=np.int32)
  #   dataset['place_yx'] = np.zeros((dataset_size, 2), dtype=np.int32)
  #   dataset['text'] = []
  #   policy = ScriptedPolicy(env)
  #   data_idx = 0
  #   while data_idx < dataset_size:
  #     np.random.seed(data_idx)
  #     num_pick, num_place = 3, 3
  #
  #     # Select random objects for data collection.
  #     pick_items = list(PICK_TARGETS.keys())
  #     pick_items = np.random.choice(pick_items, size=num_pick, replace=False)
  #     place_items = list(PLACE_TARGETS.keys())
  #     for pick_item in pick_items:  # For simplicity: place items != pick items.
  #       place_items.remove(pick_item)
  #     place_items = np.random.choice(place_items, size=num_place, replace=False)
  #     config = {'pick': pick_items, 'place': place_items}
  #
  #     # Initialize environment with selected objects.
  #     obs = env.reset(config)
  #
  #     # Create text prompts.
  #     prompts = []
  #     for i in range(len(pick_items)):
  #       pick_item = pick_items[i]
  #       place_item = place_items[i]
  #       prompts.append(f'Pick the {pick_item} and place it on the {place_item}.')
  #
  #     # Execute 3 pick and place actions.
  #     for prompt in prompts:
  #       act = policy.step(prompt, obs)
  #       dataset['text'].append(prompt)
  #       dataset['image'][data_idx, ...] = obs['image'].copy()
  #       dataset['pick_yx'][data_idx, ...] = xyz_to_pix(act['pick'])
  #       dataset['place_yx'][data_idx, ...] = xyz_to_pix(act['place'])
  #       data_idx += 1
  #       obs, _, _, _ = env.step(act)
  #       debug_clip = ImageSequenceClip(env.cache_video, fps=25)
  #       display(debug_clip.ipython_display(autoplay=1, loop=1))
  #       env.cache_video = []
  #       if data_idx >= dataset_size:
  #         break
  #
  #   pickle.dump(dataset, open(f'dataset-{dataset_size}.pkl', 'wb'))


def show_one_demonst(dataset):
  #@markdown Show a demonstration example from the dataset.
  img = dataset['image'][0]
  pick_yx = dataset['pick_yx'][0]
  place_yx = dataset['place_yx'][0]
  text = dataset['text'][0]
  plt.title(text)
  plt.imshow(img)
  plt.arrow(pick_yx[1], pick_yx[0], place_yx[1]-pick_yx[1], place_yx[0]-pick_yx[0], color='w', head_starts_at_zero=False, head_width=7, length_includes_head=True)
  plt.show()
