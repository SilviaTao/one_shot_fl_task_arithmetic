from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.eval import eval_single_dataset
import torch
from src.args import parse_arguments
from src.utils import cosine_lr, LabelSmoothing
from src.task_vectors import TaskVector
import os
import numpy as np
from src.eval import *
import numpy as np


def median_of_tvs(tvs):
  with torch.no_grad():
    new_dict = {}
    for key in tvs[0].vector:
      size = tvs[0].vector[key].size()
      value_list = torch.stack([torch.flatten(tv.vector[key]) for tv in tvs])
      median = torch.median(value_list, dim = 0)[0].reshape(size)
      new_dict[key] = median
  return TaskVector(vector = new_dict)


def fednova(tvs, local_steps):
  reweight_coeff = (np.mean(local_steps)/np.array(local_steps))/len(tvs)
  return weighted_sum_tvs(tvs, reweight_coeff)


def fedgma(tvs, rho):
  # Gradient mask average
  sign_vector = {}

  for key in tvs[0].vector:
    for tv in tvs:
      if key not in sign_vector:
        sign_vector[key] = torch.sign(tv.vector[key])
      else:
        sign_vector[key] += torch.sign(tv.vector[key])

  threshold = torch.nn.Threshold(-rho, -1, inplace = True)
  for key in sign_vector:
    sign_vector[key]  = torch.abs(sign_vector[key]/len(tvs))
    sign_vector[key] = -threshold(-sign_vector[key])
  
  sum_tvs = sum(tvs)
  new_tv_dict = {}
  for key in sign_vector:
    new_tv_dict[key] = sign_vector[key]*sum_tvs.vector[key]
  new_task_vector = TaskVector(vector = new_tv_dict)

  return new_task_vector


def cclip(tvs, rho):
  cliped_tvs = []
  for tv in tvs:
    norm = tv.norm()
    threshold = min(1, rho/norm)
    new_vector = {}
    for key in tv.vector:
      new_vector[key] = tv.vector[key]*threshold
    cliped_tvs.append(TaskVector(vector = new_vector))
  
  return sum(cliped_tvs)



def hyper_search_fedgma(task_vectors, evaluation_datasets, pretrain_checkpoint, scaling_coef_search_range, args, rho_range = np.arange(0.1, 1.0, 0.1), metric = 'normalized'): 
  
  if metric == 'normalized': # Calculate task-specific accuracy for normalization 
    task_specific_acc = {}
    for tv, eval_ds in zip(task_vectors, evaluation_datasets):
      task_specific_acc[eval_ds] = eval_single_dataset(tv.apply_to(pretrain_checkpoint, scaling_coef = 1.0), eval_ds, args)['top1'] 
  
  best_acc = 0.0
  best_rho = None
  best_scaling_coef = None
  best_model = None

  for rho in rho_range:
    curr_rho = rho
    vector = fedgma(task_vectors, curr_rho)
    for curr_coef in scaling_coef_search_range:
      print(f'rho, coef = {rho}, {curr_coef}')
      curr_acc = 0.0
      curr_model = vector.apply_to(pretrain_checkpoint, scaling_coef = curr_coef)
      for eval_ds in evaluation_datasets:
        curr_model_acc = eval_single_dataset(curr_model, eval_ds, args)['top1']
        if metric == 'normalized':
          curr_acc += curr_model_acc/(task_specific_acc[eval_ds])
  
        else:
          curr_acc += curr_model_acc
      curr_acc = curr_acc/len(evaluation_datasets)

      if curr_acc > best_acc:
          best_acc = curr_acc
          best_scaling_coef = curr_coef
          best_rho = curr_rho
          best_model = curr_model

  # Report final average test accuracy and search information
  search_info = {'scaling_coef': best_scaling_coef, 'rho': best_rho, 'metric': metric}
  print(f'search information: {search_info}')

  return best_model
    



def hyper_search_cclip(task_vectors, evaluation_datasets, pretrain_checkpoint, scaling_coef_search_range, args, number_of_rho, metric = 'normalized'):
  norm_of_tvs = [tv.norm() for tv in task_vectors]
  rho_range = np.linspace(min(norm_of_tvs), max(norm_of_tvs), number_of_rho, endpoint = False)

  
  if metric == 'normalized':
    task_specific_acc = {}
    for tv, eval_ds in zip(task_vectors, evaluation_datasets):
      task_specific_acc[eval_ds] = eval_single_dataset(tv.apply_to(pretrain_checkpoint, scaling_coef = 1.0), eval_ds, args)['top1'] 
  
  best_acc = 0.0
  best_coef = None
  best_rho = None
  best_model = None

  for rho in rho_range:
    curr_rho = rho
    for coef in scaling_coef_search_range:
      curr_coef = coef
      print(f'rho, coef = {curr_rho}, {curr_coef}')
      curr_acc = 0.0
      vector = cclip(task_vectors, curr_rho)
      curr_model = vector.apply_to(pretrain_checkpoint, scaling_coef = curr_coef)
      for eval_ds in evaluation_datasets:
        curr_model_acc = eval_single_dataset(curr_model, eval_ds, args)['top1']
        if metric == 'normalized':
          curr_acc += curr_model_acc/(task_specific_acc[eval_ds])
        else:
          curr_acc += curr_model_acc
      curr_acc = curr_acc/len(evaluation_datasets)

      if curr_acc > best_acc:
        best_acc = curr_acc
        best_rho = curr_rho
        best_coef = curr_coef
        best_model = curr_model


  search_info = {'scaling_coef': best_coef, 'rho': best_rho, 'metric': metric}  
  print(f'search information: {search_info}')
  return best_model


def scaling_coef_search(task_vectors, evaluation_datasets, pretrain_checkpoint, scaling_coef_search_range, merging_method, args, metric = 'normalized', local_steps = None):
  # This function is searching for the best coefficient used to apply task vectors to the pretrain
  # The search is conducted over validation dataset

  if metric == 'normalized':
    task_specific_acc = {}
    for tv, eval_ds in zip(task_vectors, evaluation_datasets):
      task_specific_acc[eval_ds] = eval_single_dataset(tv.apply_to(pretrain_checkpoint, scaling_coef = 1.0), eval_ds, args)['top1']

  best_acc = 0.0
  best_coef = None
  best_model = None

  # Construct merged task vector
  if merging_method == 'fednova':
    assert local_steps is not None
    vector = fednova(task_vectors, local_steps)
  elif merging_method == 'median':
    vector = median_of_tvs(task_vectors)


  for coef in scaling_coef_search_range:
    curr_coef = coef
    acc = 0.0
    model = vector.apply_to(pretrain_checkpoint, scaling_coef = curr_coef)
    for eval_ds in evaluation_datasets:
      curr_model_acc = eval_single_dataset(model, eval_ds, args)['top1']
      if metric == 'normalized':
        acc += curr_model_acc/(task_specific_acc[eval_ds])
      else:
        acc += curr_model_acc
      
      acc = acc/len(evaluation_datasets)
      if acc > best_acc:
          best_acc = acc
          best_coef = curr_coef
          best_model = model
    
  search_info = {'scaling_coef': best_coef, 'metric': metric}
  print(f'search information: {search_info}')
  return best_model


def weighted_sum_tvs(tvs, scaling_coefs):
  print('scaling coefficients are given by {}'.format(scaling_coefs))
  with torch.no_grad():
    new_dict = {}
    for key in tvs[0].vector:
      new_dict[key] = sum([tvs[i].vector[key]*scaling_coefs[i] for i in range(len(tvs))])
    
  return TaskVector(vector = new_dict)