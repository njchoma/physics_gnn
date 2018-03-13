import os
import pickle
import logging

param_name = 'model_parameters.pickle'

def init(args_in):
  loadfile = os.path.join(args_in.savedir, param_name)
  global args
  try:
    args = _load_args(loadfile)
    logging.warning("Model parameters restored from previous training")
  except:
    args = args_in
    logging.warning("Model parameters created")

def _load_args(param_file):
  with open(param_file,'rb') as filein:
    params = pickle.load(filein)
  return params

def save_args():
  savefile = os.path.join(args.savedir, param_name)
  with open(savefile, 'wb') as fileout:
    pickle.dump(args, fileout)
  logging.warning("Model parameters saved")
