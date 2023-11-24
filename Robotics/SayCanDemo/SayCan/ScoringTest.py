# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
#@title Test
import numpy as np
from const import PLACE_TARGETS, PICK_TARGETS, ENGINE
from LLMScoring import make_options, gpt3_scoring
from AffordanceScoring import affordance_scoring
from Helpers import normalize_scores
from ViLD import vild_find_object

termination_string = "done()"
query = "To pick the blue block and put it on the red block, I should:\n"

options = make_options(PICK_TARGETS, PLACE_TARGETS, termination_string=termination_string)
llm_scores, _ = gpt3_scoring(query, options, verbose=True, engine=ENGINE)

found_objects = vild_find_object()
affordance_scores = affordance_scoring(options, found_objects, block_name="box", bowl_name="circle",
                                       verbose=False, termination_string=termination_string)

combined_scores = {option: np.exp(llm_scores[option]) * affordance_scores[option] for option in options}
combined_scores = normalize_scores(combined_scores)
selected_task = max(combined_scores, key=combined_scores.get)
print("Selecting: ", selected_task)
