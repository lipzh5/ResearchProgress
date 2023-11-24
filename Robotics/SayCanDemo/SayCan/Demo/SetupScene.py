# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
#@title Setup Scene
import imageio
import numpy as np
from const import PICK_TARGETS, PLACE_TARGETS
from PickPlaceEnv import env
# import cv2
import matplotlib.pyplot as plt
from TaskConfig import config


image_path = "./2db.png"
np.random.seed(2)
if config is None:
  pick_items = list(PICK_TARGETS.keys())
  pick_items = np.random.choice(pick_items, size=np.random.randint(1, 5), replace=False)

  place_items = list(PLACE_TARGETS.keys())[:-9]
  place_items = np.random.choice(place_items, size=np.random.randint(1, 6 - len(pick_items)), replace=False)
  config = {"pick":  pick_items,
            "place": place_items}
  print(pick_items, place_items)

def init_scene():
  obs = env.reset(config)
  img_top = env.get_camera_image_top()
  # img_top_rgb = cv2.cvtColor(img_top, cv2.COLOR_BGR2RGB)
  plt.imshow(img_top)

  imageio.imsave(image_path, img_top)
  return obs