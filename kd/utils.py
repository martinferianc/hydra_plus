import os
import numpy as np
import torch
import shutil
import random
import pickle
import sys
import time
import glob
import logging
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

def config_logger(save_path):
  log_path = os.path.join(save_path, "log.log")
  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                      format=log_format, datefmt='%m/%d %I:%M:%S %p')
  fh = logging.FileHandler(log_path)
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)

def save_model(model, model_path, special_info=""):
  torch.save(model.state_dict(),
             os.path.join(model_path, f'weights{special_info}.pt'))

def save_pickle(data, path, overwrite=False):
  path = check_path(path) if not overwrite else path
  with open(path, 'wb') as fp:
      pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path):
  file = open(path, 'rb')
  return pickle.load(file)

def load_model(model, model_path):
  state_dict = torch.load(model_path, map_location=torch.device('cpu'))
  model_dict = model.state_dict()
  pretrained_dict = dict(state_dict.items())
  pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict.keys()}
  # Perform check if everything is loaded properly
  for key, value in model_dict.items():
    if key not in pretrained_dict:
        raise ValueError(f"Missing key {key} in pretrained model")
    assert value.shape == pretrained_dict[key].shape, f"Shape mismatch for key {key}"
  # Check if there are any extra keys in the pretrained model
  for key, value in pretrained_dict.items():
    if key not in model_dict:
        raise ValueError(f"Extra key {key} in pretrained model")
  model_dict.update(pretrained_dict)
  model.load_state_dict(model_dict)
    
def create_exp_dir(new_path, scripts_path):
  # Create the experiment directory and copy all .py files to it for backup
  new_path = check_path(new_path)
  os.mkdir(new_path)
  new_scripts_path = os.path.join(new_path, 'scripts')
  os.mkdir(new_scripts_path)
  for dirpath, dirnames, filenames in os.walk(scripts_path):
      structure = os.path.join(new_scripts_path, dirpath[len(scripts_path):])
      if not os.path.isdir(structure) and "__pycache__" not in dirpath:
          os.mkdir(structure)
      # Now given that we have created that directory, copy all .py files to it 
      for file in filenames:
          if file.endswith('.py') and "__pycache__" not in file:
              shutil.copy(os.path.join(dirpath, file), structure)

  # Copy all the .py files also in the current directory and save them under scripts/experiments/
  os.mkdir(os.path.join(new_path, 'scripts', 'experiments'))
  files = glob.glob('*.py')
  for file in files:
    shutil.copy(file, os.path.join(new_path, 'scripts', 'experiments'))
  
  return new_path

def check_path(path):
  if os.path.exists(path):
    filename, file_extension = os.path.splitext(path)
    counter = 0
    while os.path.exists(f'{filename}_{counter}{file_extension}'):
      counter+=1
    return f'{filename}_{counter}{file_extension}'
  return path

def model_to_gpus(model, gpu):
  if gpu>=0:
    device = torch.device(f"cuda:{str(gpu)}")
    model = model.to(device)
  return model

def decompose_experiment_name(name):
  name_parts = name.split('-')
  dataset = name_parts[0]
  method = name_parts[1]
  n_tails = int(name_parts[2])
  return dataset, method, n_tails

def parse_args(args):
  dataset = args.dataset
  method = args.method
  n_tails = args.n_tails
  new_path = '{}-{}-{}-{}'.format(dataset, method, n_tails, time.strftime("%Y%m%d-%H%M%S"))
  if args.label!="":
    new_path = '{}-{}'.format(new_path, args.label)
  new_path = create_exp_dir(
    new_path, scripts_path="../kd/")
  args.save = new_path
  
  if hasattr(args,'mute') and args.mute:
    sys.stdout = open(os.devnull, 'w')

  config_logger(save_path=args.save)

  print('Experiment dir : {}'.format(args.save))
  logging.info('Experiment dir : {}'.format(args.save))

  writer = SummaryWriter(
      log_dir=args.save+"/",max_queue=5)

  args.seed = 0 if not hasattr(args, 'seed') else args.seed
  if torch.cuda.is_available() and hasattr(args, 'gpu') and args.gpu!=-1:
    logging.info('## GPUs available = {} ##'.format(args.gpu))
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
  else:
    logging.info('## No GPUs detected ##')
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  logging.info("## Args = %s ##", args)

  path = os.path.join(args.save, 'results.pt')
  path= check_path(path)
  results = {}
  save_pickle(results, path, True)
  return args, writer

      

          
  