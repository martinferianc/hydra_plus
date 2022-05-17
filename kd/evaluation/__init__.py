import torch
import logging

from kd.evaluation.metrics import ClassificationMetric, RegressionMetric
from kd.data import options_factory, data_factory
from kd.utils import save_pickle

@torch.no_grad()
def evaluate(model, results, args):
    dataset = args.dataset
    task, _, _, levels, augmentation_types = options_factory(dataset)
    train_loader, valid_loader, test_loader = data_factory(name=dataset, batch_size=args.batch_size, random=False)
    logging.info("### Evaluating model on train ###")
    evaluate_loader(model=model, results=results, task=task, loader=train_loader, label="train", args=args)
    logging.info("### Evaluating model on valid ###")
    evaluate_loader(model=model, results=results, task=task, loader=valid_loader, label="valid", args=args)
    logging.info("### Evaluating model on test ###")
    test_metric = evaluate_loader(model=model, results=results, task=task, loader=test_loader, label="test", args=args)
    for i, augmentation in enumerate(augmentation_types):
        for level in levels[i]:
            _, _, test_loader = data_factory(name=dataset, batch_size=args.batch_size, random=True, augmentation=augmentation, level=level)
            logging.info("### Evaluating model on test augmentation: {} level: {} ###".format(augmentation, level))
            evaluate_loader(model=model, results=results, task=task, loader=test_loader, label="{}_{}".format(augmentation, level), args=args)
    return results, test_metric

def evaluate_loader(model, results, task, loader, label, args):
    if task == "classification":
        metric = ClassificationMetric(None, model.output_size)  
    elif task == "regression":
        metric = RegressionMetric(None)
    metric, outputs = step(loader, model, metric, args.debug)
    metric = record_metrics_and_outputs(metric, outputs, results, label, args.save)
    return metric

def step(loader, model, metric, debug=False):
    outputs = []
    for i, (input, target) in enumerate(loader):
      if next(model.parameters()).is_cuda:
        input = input.to(next(model.parameters()).device)
        target = target.to(next(model.parameters()).device)
        
      output = model(input)
      outputs.append(output.detach().cpu())
      if i >= 2 and debug:
          break
      metric.update(output, target)

      if debug:
        break
    
    outputs = torch.cat(outputs, dim=0)  
    return metric, outputs

def record_metrics_and_outputs(metric, outputs, results, label, path=None):
    metrics = metric.get_packed()
    for key, value in metrics.items():
        logging.info("### Result metric {}: {}: {} ###".format(label, key, value))
        if results is not None:
          if key not in results:
            results[key] = {}
          results[key][label] = value
    if path is not None:
      save_pickle(outputs, "{}/{}_outputs.pt".format(path, label))
    return metrics
