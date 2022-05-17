import torch
import logging

from kd.training.trainer import Trainer
from kd.training.utils import WeightDecay, GradientClipping
from kd.data import options_factory, data_factory
from kd.training.losses import losses_factory

def train(args, model, teacher=None, writer=None, special_id="", hyperparameters={}):
    dataset = args.dataset
    batch_size = args.batch_size
    method = args.method
    task, _, _, _, _ = options_factory(dataset)
    train_time, valid_time = None, None

    logging.info("## Start training: Dataset: %s, Method: %s, Task: %s ##" % (dataset, method, task))
    logging.info("## Hyperparameters: %s ##" % hyperparameters)

    logging.info('### Preparing criterion ###')
    criterion = losses_factory(task=task, method=method)
    logging.info('### Preparing schedulers and optimizers ###')
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["lr"], weight_decay=0.0)
    wd = WeightDecay(hyperparameters["wd"])
    grad_clip = GradientClipping(hyperparameters["grad_clip"])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, hyperparameters["epochs"])
    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, epochs=hyperparameters["epochs"], task=task, teacher=teacher, writer=writer,\
                    temperature_mean_scheduler=hyperparameters["temperature_mean_scheduler"], temperature_individual_scheduler=hyperparameters["temperature_individual_scheduler"],  
                    kl_scheduler=hyperparameters["kl_scheduler"], wd = wd, grad_clip=grad_clip, args=args,
                    lambda_scheduler=hyperparameters["lambda_scheduler"], verbose=True, lr_scheduler=lr_scheduler,
                    alpha_scheduler=hyperparameters["alpha_scheduler"], beta_scheduler=hyperparameters["beta_scheduler"])
    logging.info('### Downloading and preparing data ##')
    train_loader, valid_loader, _ = data_factory(dataset, batch_size=batch_size, random=False)
    train_time, valid_time = trainer.train_loop(train_loader, valid_loader, special_id)

    logging.info("## End training, took: %.2f seconds ##" % (train_time + valid_time))
    return model

    
