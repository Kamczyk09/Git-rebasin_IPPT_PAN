import copy
import numpy as np

def flatten_params(model):
  return model.state_dict()

def lerp(lam, t1, t2):
  t3 = copy.deepcopy(t2)
  for p in t1: 
    t3[p] = (1 - lam) * t1[p] + lam * t2[p]
  return t3


def merge_models(model1, model2, alpha=0.5):
  assert type(model1) == type(model2), "Models need to be of same type"
  merged_model = copy.deepcopy(model1)
  state_dict1 = model1.state_dict()
  state_dict2 = model2.state_dict()
  merged_state_dict = {}

  for key in state_dict1:
    merged_state_dict[key] = alpha * state_dict1[key] + (1 - alpha) * state_dict2[key]

  merged_model.load_state_dict(merged_state_dict)
  return merged_model

