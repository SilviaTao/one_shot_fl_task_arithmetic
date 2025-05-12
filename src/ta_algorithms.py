
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.eval import eval_single_dataset
import torch
from src.args import parse_arguments
from src.utils import cosine_lr, LabelSmoothing
from src.task_vectors import TaskVector
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
from src.helpers import *
from src.eval import *
import numpy as np
from src.utils import WORK_DIR


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


def fedgma(tvs, tau):
  # Gradient mask average
  sign_vector = {}

  for key in tvs[0].vector:
    for tv in tvs:
      if key not in sign_vector:
        sign_vector[key] = torch.sign(tv.vector[key])
      else:
        sign_vector[key] += torch.sign(tv.vector[key])

  threshold = torch.nn.Threshold(-tau, -1, inplace = True)
  for key in sign_vector:
    sign_vector[key]  = torch.abs(sign_vector[key]/len(tvs))
    sign_vector[key] = -threshold(-sign_vector[key])
  
  sum_tvs = sum(tvs)
  new_tv_dict = {}
  for key in sign_vector:
    new_tv_dict[key] = sign_vector[key]*sum_tvs.vector[key]
  new_task_vector = TaskVector(vector = new_tv_dict)

  return new_task_vector


def hyper_search_fedgma(tvs, datasets, evaluation_datasets, pretrain, args, search_range, tau_range = np.arange(0.1, 1.0, 0.1), metric = 'normalized'):
  print(tau_range)  
  task_specific_acc = {}
  if metric == 'normalized':
    for tv, eval_ds in zip(tvs, evaluation_datasets):
      task_specific_acc[eval_ds] = eval_single_dataset(tv.apply_to(pretrain, scaling_coef = 1.0), eval_ds, args)['top1'] 
  
  best_acc = 0.0
  best_tau = None
  best_scaling_coef = None
  best_model = None

  for tau in tau_range:
    curr_tau = tau
    vector = fedgma(tvs, curr_tau)
    for curr_coef in search_range:
      print(f'tau, coef = {tau}, {curr_coef}')
      curr_acc = 0.0
      curr_model = vector.apply_to(pretrain, scaling_coef = curr_coef)
      for ds in datasets:
        curr_model_acc = eval_single_dataset(curr_model, ds, args)['top1']
        if metric == 'normalized':
          curr_acc += curr_model_acc/(task_specific_acc[ds.split('Val')[0]])
  
        else:
          curr_acc += curr_model_acc
      curr_acc = curr_acc/len(datasets)

      if curr_acc > best_acc:
          best_acc = curr_acc
          best_scaling_coef = curr_coef
          best_tau = curr_tau
          best_model = curr_model

      
    search_info = {'scaling_coef': best_scaling_coef, 'tau': best_tau, 'metric': metric}
    evaluation_avg_acc = 0.0
    for eval_ds in evaluation_datasets:
      best_model_acc = eval_single_dataset(best_model, eval_ds, args)['top1']
      if metric == 'normalized':
        search_info[eval_ds] = best_model_acc/(task_specific_acc[eval_ds])
      else:
        search_info[eval_ds] = best_model_acc
      evaluation_avg_acc += search_info[eval_ds]
    evaluation_avg_acc = evaluation_avg_acc/len(evaluation_datasets)
    
    return evaluation_avg_acc, search_info
    

def cclip(tvs, norm_of_tvs, tau):
  cliped_tvs = []
  for tv, norm in zip(tvs, norm_of_tvs):
    threshold = min(1, tau/norm)
    new_vector = {}
    for key in tv.vector:
      new_vector[key] = tv.vector[key]*threshold
    cliped_tvs.append(TaskVector(vector = new_vector))
  
  return sum(cliped_tvs)


def hyper_search_cclip(tvs, datasets, evaluation_datasets, pretrain, args, search_range, metric = 'normalized'):
  norm_of_tvs = [tv.norm().cpu() for tv in tvs]
  tau_range = np.linspace(min(norm_of_tvs), max(norm_of_tvs), 5, endpoint = False)

  task_specific_acc = {}
  if metric == 'normalized':
    for tv, eval_ds in zip(tvs, evaluation_datasets):
      task_specific_acc[eval_ds] = eval_single_dataset(tv.apply_to(pretrain, scaling_coef = 1.0), eval_ds, args)['top1'] 
  
  best_acc = 0.0
  best_coef = None
  best_tau = None
  best_model = None

  for tau in tau_range:
    curr_tau = tau
    for coef in search_range:
      curr_coef = coef
      print(f'tau, coef = {curr_tau}, {curr_coef}')
      curr_acc = 0.0
      vector = cclip(tvs, norm_of_tvs, curr_tau)
      curr_model = vector.apply_to(pretrain, scaling_coef = curr_coef)
      for ds in datasets:
        curr_model_acc = eval_single_dataset(curr_model, ds, args)['top1']
        if metric == 'normalized':
          curr_acc += curr_model_acc/(task_specific_acc[ds.split('Val')[0]])
        else:
          curr_acc += curr_model_acc
      curr_acc = curr_acc/len(datasets)

      if curr_acc > best_acc:
        best_acc = curr_acc
        best_tau = curr_tau
        best_coef = curr_coef
        best_model = curr_model


  search_info = {'scaling_coef': best_coef, 'tau': best_tau, 'metric': metric}
  
  evaluation_avg_acc = 0.0
  for eval_ds in evaluation_datasets:
    best_model_acc = eval_single_dataset(best_model, eval_ds, args)['top1']
    if metric == 'normalized':
      search_info[eval_ds] = best_model_acc/(task_specific_acc[eval_ds])
    else:
      search_info[eval_ds] = best_model_acc
    evaluation_avg_acc += search_info[eval_ds]
  evaluation_avg_acc = evaluation_avg_acc/len(evaluation_datasets)
  
  return evaluation_avg_acc, search_info


def coef_search(tvs, vector, datasets, evaluation_datasets, pretrain, args, search_range, metric = 'normalized'):
    # This function is searching for the best coefficient used to apply task vectors to the pretrain
    # The search is conducted over validation dataset
    task_specific_acc = {}
    if metric == 'normalized':
      for tv, eval_ds in zip(tvs, evaluation_datasets):
        task_specific_acc[eval_ds] = eval_single_dataset(tv.apply_to(pretrain, scaling_coef = 1.0), eval_ds, args)['top1']
    
    curr_coef = search_range[0]
    curr_acc = 0.0
    curr_model = vector.apply_to(pretrain, scaling_coef = curr_coef)
    for ds in datasets:
      curr_model_acc = eval_single_dataset(curr_model, ds, args)['top1']
      if metric == 'normalized':

        curr_acc += curr_model_acc/(task_specific_acc[ds.split('Val')[0]])
  
      else:
        curr_acc += curr_model_acc
    curr_acc = curr_acc/len(datasets)

    for i in range(1, len(search_range)):
        coef = search_range[i]
        acc = 0.0
        model = vector.apply_to(pretrain, scaling_coef = coef)
        for ds in datasets:
          curr_model_acc = eval_single_dataset(model, ds, args)['top1']
          if metric == 'normalized':

            acc += curr_model_acc/(task_specific_acc[ds.split('Val')[0]])
          else:
            acc += curr_model_acc
        acc = acc/len(datasets)
        if acc > curr_acc:
            curr_acc = acc
            curr_coef = coef
            curr_model = model
      
    search_info = {'scaling_coef': curr_coef, 'metric': metric}
    evaluation_avg_acc = 0.0
    for eval_ds in evaluation_datasets:
      curr_model_acc = eval_single_dataset(curr_model, eval_ds, args)['top1']
      if metric == 'normalized':
        search_info[eval_ds] = curr_model_acc/(task_specific_acc[eval_ds])
      else:
        search_info[eval_ds] = curr_model_acc
      evaluation_avg_acc += search_info[eval_ds]
    evaluation_avg_acc = evaluation_avg_acc/len(evaluation_datasets)
    
    return evaluation_avg_acc, curr_model, search_info
