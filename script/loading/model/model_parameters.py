import os
import pickle
import logging

param_name = 'model_parameters.pickle'

def init(args_in):
  loadfile = os.path.join(args_in.savedir, param_name)
  global args
  try:
    args = _load_args(loadfile)
    logging.warning("Model arguments restored from previous training")
    # Update run-specific arguments
    args.cuda = args_in.cuda
    args.plot = args_in.plot
    args.nbtrain  = args_in.nbtrain
    args.nbtest   = args_in.nbtest
    args.nbprint  = args_in.nbprint
    args.nb_batch = args_in.nb_batch
    args.shuffle_while_training = args_in.shuffle_while_training
    args.sorted_training = args_in.sorted_training
  except:
    args = args_in
    logging.warning("Model arguments created")

def _load_args(param_file):
  with open(param_file,'rb') as filein:
    params = pickle.load(filein)
  return params

def save_args():
  savefile = os.path.join(args.savedir, param_name)
  with open(savefile, 'wb') as fileout:
    pickle.dump(args, fileout)
  logging.warning("Model arguments saved")
