# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
#@title Runner
import time

import numpy as np
from const import PICK_TARGETS, PLACE_TARGETS, ENGINE, CATEGORY_NAMES
from PickPlaceEnv import env
import SayCan
from SayCan.AffordanceScoring import affordance_scoring
from SayCan.LLMScoring import make_options, gpt3_scoring
from SayCan.Helpers import build_scene_description, normalize_scores, normalize_scores, plot_saycan, step_to_nlp
import matplotlib.pyplot as plt
from DemoSetup import gpt3_context, use_environment_description
from TaskConfig import raw_input, only_plan
from DemoSetup import termination_string, vild_params
from SetupScene import image_path, init_scene
from ViLD import ViLD
from CliportDemo import run_cliport


def run():
  obs = init_scene()  # Note: should init scene fitst (save img at this stage)
  plot_on = True
  max_tasks = 3
  options = make_options(PICK_TARGETS, PLACE_TARGETS, termination_string=termination_string)
  # ViLD (open vocabulary object detection)
  category_name_string = ";".join(CATEGORY_NAMES)
  found_objects = ViLD.vild(image_path, category_name_string, vild_params, plot_on=False)
  print('vild found objects: ', found_objects)
  scene_description = build_scene_description(found_objects)
  env_description = scene_description

  print(scene_description)

  gpt3_prompt = gpt3_context
  if use_environment_description:
    gpt3_prompt += "\n" + env_description
  gpt3_prompt += "\n# " + raw_input + "\n"

  all_llm_scores = []
  all_affordance_scores = []
  all_combined_scores = []
  # calculate all the option scores according to the obj detection res
  affordance_scores = affordance_scoring(options, found_objects, block_name="box", bowl_name="circle", verbose=False)
  for option, score in affordance_scores.items():
    print(option, score)
  num_tasks = 0
  selected_task = ""
  steps_text = []
  while not selected_task == termination_string:
    num_tasks += 1
    if num_tasks > max_tasks:
      break
    #time.sleep(60)
    # completions api
    llm_scores, _ = gpt3_scoring(gpt3_prompt, options, verbose=True, engine=ENGINE, print_tokens=False)
    combined_scores = {option: np.exp(llm_scores[option]) * affordance_scores[option] for option in options}
    combined_scores = normalize_scores(combined_scores)
    selected_task = max(combined_scores, key=combined_scores.get)
    steps_text.append(selected_task)
    print('num tasks: ', num_tasks, "Selecting: ", selected_task)
    gpt3_prompt += selected_task + "\n"     # append selection to the prompt

    all_llm_scores.append(llm_scores)
    all_affordance_scores.append(affordance_scores)
    all_combined_scores.append(combined_scores)

  if plot_on:
    for llm_scores, affordance_scores, combined_scores, step in zip(
        all_llm_scores, all_affordance_scores, all_combined_scores, steps_text):
      plot_saycan(llm_scores, affordance_scores, combined_scores, step, show_top=10)

  print('**** Solution ****')
  print(env_description)
  print('# ' + raw_input)
  for i, step in enumerate(steps_text):
    if step == '' or step == termination_string:
      break
    print('Step ' + str(i) + ': ' + step)
    nlp_step = step_to_nlp(step)

  if not only_plan:
    print('Initial state:')
    plt.imshow(env.get_camera_image())
    plt.show()

    for i, step in enumerate(steps_text):
      if step == '' or step == termination_string:
        break
      nlp_step = step_to_nlp(step)
      print('GPT-3 says next step:', nlp_step)

      obs = run_cliport(obs, nlp_step)

    # Show camera image after task.
    print('Final state:')
    plt.imshow(env.get_camera_image())
    plt.show()


if __name__ == "__main__":
  # init_scene()
  # camera_img = env.get_camera_image()
  # # print('camera img: ', camera_img)
  #
  # plt.imshow(camera_img)
  # plt.show()
  # options = make_options(PICK_TARGETS, PLACE_TARGETS, termination_string=termination_string)
  # print('runner options: ', options)
  run()
