import sys
import argparse
import logging
import json 

sys.path.append("../")
from kd.evaluation import evaluate
from kd.models import model_factory
import kd.utils as utils
from kd.evaluation.profile import profile_model
from kd.training.hyperparameters import get_hyperparameters

def evaluate_model(args, writer, model=None):
  logging.info('# Starting Evaluation #')
  if model is None:
    hyperparameters = get_hyperparameters(args.dataset, args.method, args.n_tails)
    if args.method == "ensemble":
      models = []
      logging.info('## Creating Ensemble Model ##')
      for i in range(args.n_tails):
        logging.info('## Creating Individual: %d ##' % i)
        model = model_factory(dataset=args.dataset, method=args.method, n_tails=1, models=None, train=True, hyperparameters=hyperparameters)
        models.append(model)
      model = model_factory(dataset=args.dataset, method=args.method, n_tails=args.n_tails, models=models, train=False, hyperparameters=hyperparameters)
    else:
      logging.info('## Creating Model ##')
      model = model_factory(dataset=args.dataset, method=args.method, n_tails=args.n_tails, models=None, train=False, hyperparameters=hyperparameters)
    logging.info('## Model created: ##')
    logging.info(model.__repr__())
    # This is with respect to loading the default model after training
    utils.load_model(model, args.load if hasattr(args, 'load') else args.save+"/weights.pt")

  model = utils.model_to_gpus(model, args.gpu)

  logging.info('## Evaluating ##')
  results = utils.load_pickle(args.save+"/results.pt")
  logging.info('### Setting the model to evaluation mode ###')
  model.eval()
  if args.method == "endd":
      # This is to guarantee that n_predictions are run if the model is passed directly from training 
      model.tail_model.n_predictions = args.n_tails

  evaluate(model, results, args)

  logging.info('## Evaluation Done ##')

  logging.info("### Profiling ###")
  profile_model(model, results, args)
  logging.info("### Finished Profiling ###")

  utils.save_pickle(results, args.save+"/results.pt",overwrite=True)

if __name__ == '__main__':
  parser = argparse.ArgumentParser("evaluation")

  parser.add_argument('--dataset', type=str, default='spiral', help='define the dataset and thus task')
  parser.add_argument('--method', type=str, default='ensemble', help='what method to train')               

  parser.add_argument('--batch_size', type=int, default=256, help='batch size')
  parser.add_argument('--n_tails', type=int, default=10, help='random seed')

  parser.add_argument('--save', type=str, default='EXP', help='experiment name')
  parser.add_argument('--label', type=str, default='', help='experiment name')
  parser.add_argument('--load', type=str, default='EXP', help='experiment name')

  parser.add_argument('--seed', type=int, default=1, help='random seed')
  parser.add_argument('--debug', action='store_true', help='whether we are currently debugging')
  parser.add_argument('--mute', action='store_true', help='whether we are currently debugging')
  parser.add_argument('--gpu', type=int, default = 0, help='gpu device ids')
  parser.add_argument('--hyperparameters', type=json.loads, default={})

  args = parser.parse_args()
  args, writer = utils.parse_args(args)
  evaluate_model(args, writer)