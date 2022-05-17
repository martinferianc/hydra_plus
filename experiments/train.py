import sys
import argparse
import logging
import json 

sys.path.append("../")
from kd.training import train
from evaluate import evaluate_model
from kd.models import model_factory
from kd.training.hyperparameters import get_hyperparameters
import kd.utils as utils

def train_model(args, writer):
  logging.info('# Starting training #')
  model = None
  hyperparameters = get_hyperparameters(args.dataset, args.method, args.n_tails, args.hyperparameters)
  if args.method == "ensemble":
    models = []
    logging.info('## Training Ensemble ##')
    for i in range(args.n_tails):
      logging.info('## Training Individual: %d ##' % i)
      model = model_factory(dataset=args.dataset, method=args.method, n_tails=1, models=None, train=True, hyperparameters=hyperparameters)
      logging.info(model.__repr__())
      model = utils.model_to_gpus(model, args.gpu)
      train(args, model, teacher=None, writer=writer, special_id=i, hyperparameters=hyperparameters)
      models.append(model.cpu())
    logging.info('## Creating Ensemble ##')
    model = model_factory(dataset=args.dataset, method=args.method, n_tails=args.n_tails, models=models, train=False, hyperparameters=hyperparameters)
    logging.info(model.__repr__())
    logging.info('## Created Ensemble ##')
  else:
    teacher_model = None
    if args.load_teacher!="":
      empty_models = []
      hyperparameters_teacher = get_hyperparameters(dataset=args.dataset, method="ensemble", n_tails=-1, overwrite_hyperparameters={})
      for _ in range(args.n_tails_teacher):
        model = model_factory(dataset=args.dataset, method="ensemble", n_tails=1, models=None, train=True,hyperparameters=hyperparameters_teacher)
        empty_models.append(model)
      logging.info('## Loading Teacher ##')
      teacher_model = model_factory(dataset=args.dataset, method="ensemble", n_tails=args.n_tails_teacher, models=empty_models, train=False, hyperparameters=hyperparameters_teacher)
      utils.load_model(teacher_model, args.load_teacher)
      logging.info('## Teacher: {} loaded ##'.format(args.load_teacher))
      logging.info(teacher_model.__repr__())
      teacher_model = utils.model_to_gpus(teacher_model, args.gpu)
    
    logging.info('## Loading main model ##')
    model = model_factory(args.dataset, args.method, args.n_tails, None, train=True, hyperparameters=hyperparameters)
    model = utils.model_to_gpus(model, args.gpu)
    logging.info('## Model created: ##')
    logging.info(model.__repr__())
    train(args, model, teacher=teacher_model, writer=writer, hyperparameters=hyperparameters)
  

  utils.save_model(model, args.save)

  logging.info('# Finished training #')
  logging.info('# Evaluating with respect to default parameters during training #')
  evaluate_model(args, writer, model)

if __name__ == '__main__':
  parser = argparse.ArgumentParser("training")

  parser.add_argument('--dataset', type=str, default='spiral', help='define the dataset and thus task')
  parser.add_argument('--method', type=str, default='teacher', help='what method to train')               

  parser.add_argument('--batch_size', type=int, default=256, help='batch size')
  parser.add_argument('--n_tails', type=int, default=10, help='random seed')

  parser.add_argument('--save', type=str, default='EXP', help='experiment name')
  parser.add_argument('--label', type=str, default='', help='experiment name')
  parser.add_argument('--load_teacher', type=str, default='', help='experiment name')
  parser.add_argument('--n_tails_teacher', type=int, default=10, help='random seed')

  parser.add_argument('--seed', type=int, default=1, help='random seed')
  parser.add_argument('--debug', action='store_true', help='whether we are currently debugging')
  parser.add_argument('--mute', action='store_true', help='whether we are currently debugging')
  parser.add_argument('--gpu', type=int, default = 0, help='gpu device ids')
  parser.add_argument('--hyperparameters', type=json.loads, default={})

  
  args = parser.parse_args()
  args, writer = utils.parse_args(args)
  train_model(args, writer)