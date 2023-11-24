# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
# setup environment
# define PyBullet-based environment with a UR5e and 2-finger gripper
import numpy as np

PICK_TARGETS = {
    'blue block': None,
    'red block': None,
    'green block': None,
    'yellow block': None,
}

COLORS = {
    'blue':(78/255, 121/255, 167/255, 255/255),
    'red':(255/255, 87/255, 89/255, 255/255),
    'green': (89/255, 169/255, 79/255, 255/255),
    'yellow': (237/255, 201/255, 72/255, 255/255),
}

PLACE_TARGETS = {
    'blue block': None,
    'red block': None,
    'green block': None,
    'yellow block': None,

    'blue bowl': None,
    'red bowl': None,
    'green bowl': None,
    'yellow bowl': None,
    # (x, y, z)
    'top left corner': (-0.3 + 0.05, -0.2 - 0.05, 0),
    'top right corner': (0.3 + 0.05, -0.2 - 0.05, 0),
    'middle': (0, -0.5, 0),
    'bottom left corner': (-0.3 + 0.05, -0.8 + 0.05, 0),
    'bottom right corner': (0.3 - 0.05, -0.8 + 0.05, 0),
}
CATEGORY_NAMES = ['blue block',
					  'red block',
					  'green block',
					  'orange block',
					  'yellow block',
					  'purple block',
					  'pink block',
					  'cyan block',
					  'brown block',
					  'gray block',

					  'blue bowl',
					  'red bowl',
					  'green bowl',
					  'orange bowl',
					  'yellow bowl',
					  'purple bowl',
					  'pink bowl',
					  'cyan bowl',
					  'brown bowl',
					  'gray bowl']

PIXEL_SIZE = 0.00267857
BOUNDS = np.float32([[-0.3, 0.3], [-0.8, -0.2], [0, 0.15]]) # X Y Z

openai_api_key = "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"
ENGINE =  "text-davinci-002"  # "gpt-3.5-turbo-instruct" #
#"text-ada-001" # "text-davinci-002"
