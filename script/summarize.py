import os
import argparse
import numpy as np

'''
Summarize all models in a folder if not already completed
Good for array runs 0-99
'''

def get_one_model_stats(model_dir):
  '''
  Get statistics of one trained model
  Best model chosen by test 1/FPR
  Can add more statistics here if desired
  '''
  csv_file = [f for f in os.listdir(model_dir) if f.endswith(".csv")][0]
  csv_file = os.path.join(model_dir, csv_file)
  with open(csv_file, 'r') as filein:
    text = np.loadtxt(filein, delimiter=",", skiprows=1)
  best_run_idx = np.argmax(text[:,6]) # Get best test FPR
  best_run = text[best_run_idx]
  stats = {}
  stats['test_auc'] = best_run[4]
  stats['test_1/fpr'] = best_run[6]
  stats['nb_epoch'] = text.shape[0]
  stats['best_epoch'] = best_run_idx+1 # Epoch number starts at 1
  return stats

def write_one_model(f, model):
  '''
  Write summary for one trained model
  '''
  f.write(model['name']+'\n')
  for key in model:
    if key == 'name':
      continue
    f.write("{0:.6g}  ".format(model[key]))
    f.write("{}\n".format(key))
  f.write("\n")

def summarize_stats(stats_all_models):
  '''
  Get mean and standard error for each statistic
  '''
  mean = {}
  err = {}
  nb_models = len(stats_all_models)

  # Get mean
  for key in stats_all_models[0]:
    if type(stats_all_models[0][key]) == str:
      continue
    mean[key] = 0
    for m in stats_all_models:
      mean[key] += m[key]
    mean[key] /= nb_models

  # Get std err of mean
  for key in stats_all_models[0]:
    if type(stats_all_models[0][key]) == str:
      continue
    err[key] = 0
    for m in stats_all_models:
      sqdiff = (mean[key]-m[key])**2
      err[key] += sqdiff / nb_models
    err[key] = np.sqrt(err[key] / nb_models)
  return mean, err

def summarize_one_model_type(models_dir, model_name):
  '''
  Summarize statistics of all models which begin with model_name
  '''
  models = [os.path.join(models_dir, o) for o in os.listdir(models_dir)
                                    if o[:-2] == model_name]
  model_info = []
  stats_mean = {}
  for m in models:
    try:
      stats = get_one_model_stats(m)
      stats['name'] = os.path.basename(m)
      model_info.append(stats)
    except:
      continue
  stats_mean, stats_stderr = summarize_stats(model_info)
  nb_models = len(model_info)
  filename_out = os.path.join(models_dir, model_name) + '.txt'
  # Write summary statistics for all models combined
  with open(filename_out, 'w') as f:
    f.write("SUMMARY:\n")
    f.write("{} models trained\n".format(nb_models))
    for key in stats_mean:
      f.write("{0:.6g} ".format(stats_mean[key]))
      f.write(" +/- {0:.3g}  ".format(stats_stderr[key]))
      f.write("{}\n".format(key))
    f.write("\n\n")
    # Write statistics for each trained model
    for m in model_info:
      write_one_model(f,m)

def summarize_all_in_dir(models_dir):
  '''
  Finds all models within the selected directory and
  performs summary of the relevant statistics
  '''
  model_folders = [o for o in os.listdir(models_dir) 
                      if os.path.isdir(os.path.join(models_dir,o))]
  unique_names = set()
  to_summarize = set()
  for model in model_folders:
    mdl_name = model[:-2]
    if mdl_name in unique_names:
      to_summarize.add(mdl_name)
    unique_names.add(mdl_name)
  for model in to_summarize:
    try:
      summarize_one_model_type(models_dir, model)
    except:
      print("Issue with {}".format(model))
      continue

def read_args():
  parser = argparse.ArgumentParser(description='Module to summarize training GCNN training runs')
  add_arg = parser.add_argument

  add_arg('--path',dest='path',help='File path to trained models directory',required=True)
  return parser.parse_args()

if __name__ == "__main__":
  args = read_args()
  summarize_all_in_dir(args.path)
