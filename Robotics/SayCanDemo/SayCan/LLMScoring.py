# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None

import openai
from const import PICK_TARGETS, PLACE_TARGETS, ENGINE

#@title LLM Cache
overwrite_cache = True
if overwrite_cache:
  LLM_CACHE = {}
# openai.api_key = "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"
openai.api_key = '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'
openai.api_key = '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'

#@title LLM Scoring

def gpt3_call(engine="text-ada-001", prompt="", max_tokens=128, temperature=0,
              logprobs=1, echo=False):
  full_query = ""
  for p in prompt:
    full_query += p
  print('full query: ', full_query)
  print('gpt3 prompt: ', prompt)
  id = tuple((engine, full_query, max_tokens, temperature, logprobs, echo))
  if id in LLM_CACHE.keys():
    response = LLM_CACHE[id]
  else:
    response = openai.Completion.create(engine=engine,
                                        prompt=prompt,
                                        max_tokens=max_tokens,
                                        temperature=temperature,
                                        logprobs=logprobs,
                                        echo=echo)
    LLM_CACHE[id] = response
  return response

def gpt3_scoring(query, options, engine="text-ada-001", limit_num_options=None, option_start="\n", verbose=False, print_tokens=False):
  if limit_num_options:
    options = options[:limit_num_options]
  verbose and print("Scoring", len(options), "options")
  gpt3_prompt_options = [query + option for option in options]
  # query+option
  response = gpt3_call(
      engine=engine,
      prompt=gpt3_prompt_options,
      max_tokens=0,
      logprobs=1,
      temperature=0,
      echo=True,)
  # print('gpt3 prompt options: ', gpt3_prompt_options)
  # print('responses text: ', type(response['choices']))
  # print('choices keys: ', response['choices'][0].keys())

  # for choice in response['choices']:
  #   print('choices text: ', choice['text'])
  #   print('==='*5)
  # print('%d choices in total: ' % len(response['choices']))
  # print('responses logporb: ', response['choices'])

  scores = {}
  for option, choice in zip(options, response["choices"]):
    tokens = choice["logprobs"]["tokens"]
    token_logprobs = choice["logprobs"]["token_logprobs"]

    total_logprob = 0
    for token, token_logprob in zip(reversed(tokens), reversed(token_logprobs)):
      print_tokens and print(token, token_logprob)
      if option_start is None and not token in option:
        break
      if token == option_start:
        break
      total_logprob += token_logprob
    scores[option] = total_logprob

  for i, option in enumerate(sorted(scores.items(), key=lambda x : -x[1])):
    verbose and print(option[1], "\t", option[0])
    if i >= 10:
      break

  return scores, response

def make_options(pick_targets=None, place_targets=None, options_in_api_form=True, termination_string="done()"):
  if not pick_targets:
    pick_targets = PICK_TARGETS
  if not place_targets:
    place_targets = PLACE_TARGETS
  options = []
  for pick in pick_targets:
    for place in place_targets:
      if options_in_api_form:
        option = "robot.pick_and_place({}, {})".format(pick, place)
      else:
        option = "Pick the {} and place it on the {}.".format(pick, place)
      options.append(option)

  options.append(termination_string)
  print("Considering", len(options), "options")
  return options

query = "To pick the blue block and put it on the red block, I should:\n"
options = make_options(PICK_TARGETS, PLACE_TARGETS)
scores, response = gpt3_scoring(query, options, engine=ENGINE, limit_num_options=5, option_start='\n', verbose=True)

if __name__ == "__main__":
  query = "To pick the blue block and put it on the red block, I should:\n"
  options = make_options(PICK_TARGETS, PLACE_TARGETS)
  print('options: ', options)
  print('pick targets: ', PLACE_TARGETS, PLACE_TARGETS)
  print('query: ', query)
  scores, response = gpt3_scoring(query, options, engine=ENGINE, limit_num_options=5, option_start='\n', verbose=True)
  #print('scores: ', scores)
  for option, score in scores.items():
    print(option, score)
  print('responses: ', response['choices'][0]["logprobs"]["token_logprobs"])
  print('len out tokens prob: ', len(response['choices'][0]["logprobs"]["token_logprobs"]))
  print('choices: ', response['choices'][0]['logprobs'].keys())
  print('len tokens: ', len(response['choices'][0]['logprobs']['tokens']))
  print('tokens: ', response['choices'][0]['logprobs']['tokens'])



