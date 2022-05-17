import torch
import time
import logging

from kd.evaluation.metrics import ClassificationMetric, RegressionMetric
from kd.training.utils import LinearScheduler

class Trainer():
  def __init__(self, model, 
                     criterion, 
                     optimizer, 
                     epochs,
                     wd,
                     grad_clip, 
                     
                     alpha_scheduler = 0.9,
                     beta_scheduler = 0.0,
                     lr_scheduler=None,
                     kl_scheduler=None,
                     lambda_scheduler=None,
                     temperature_mean_scheduler=None,
                     temperature_individual_scheduler=None,

                     writer=None, 
                     args=None, 
                     teacher=None, 
                     verbose=False, 
                     task="classification",
                     logging_frequency=50,
                     weight_logging_frequency=-1):
    
    super().__init__()
    self.model = model
    self.criterion = criterion
    self.optimizer = optimizer
    self.epochs = epochs
    self.wd = wd
    self.grad_clip = grad_clip

    self.alpha_scheduler = alpha_scheduler
    self.beta_scheduler = beta_scheduler
    self.lr_scheduler = lr_scheduler
    self.kl_scheduler = kl_scheduler
    self.lambda_scheduler = lambda_scheduler
    self.temperature_mean_scheduler = temperature_mean_scheduler
    self.temperature_individual_scheduler = temperature_individual_scheduler
    self.grad_clip_coeff = grad_clip.coeff
    self.wd_coeff = self.wd.coeff

    self.writer = writer
    self.args = args
    self.teacher = teacher
    self.verbose = verbose
    self.logging_frequency = logging_frequency
    self.weight_logging_frequency = weight_logging_frequency

    self.train_time = 0.0
    self.val_time = 0.0

    if task == "classification":
      self.train_metrics = ClassificationMetric(writer, model.output_size)
      self.valid_metrics = ClassificationMetric(writer, model.output_size)
      
    elif task == "regression":
      self.train_metrics = RegressionMetric(writer)
      self.valid_metrics = RegressionMetric(writer)
      
    assert task in ["classification", "regression"]

    self.writer = writer
    self.epoch = -1
    self.iteration = 0

    if self.teacher is not None:
      self.teacher.eval()

  def train_loop(self, train_loader, valid_loader, special_id=""):
    for epoch in range(self.epochs):
      self.epoch = epoch
      if epoch >= 1 and self.lr_scheduler is not None:
          self.lr_scheduler.step()

      if epoch>=1 and self.kl_scheduler is not None and not isinstance(self.kl_scheduler, float):
          self.kl_scheduler.step()

      if epoch>=1 and self.lambda_scheduler is not None and not isinstance(self.lambda_scheduler, float):
          self.lambda_scheduler.step()

      if epoch >= 1 and self.temperature_mean_scheduler is not None and not isinstance(self.temperature_mean_scheduler, float):
          self.temperature_mean_scheduler.step()

      if epoch >= 1 and self.temperature_individual_scheduler is not None and not isinstance(self.temperature_individual_scheduler, float):
          self.temperature_individual_scheduler.step()

      if epoch >= 1 and self.alpha_scheduler is not None and not isinstance(self.alpha_scheduler, float):
          self.alpha_scheduler.step()

      if epoch >= 1 and self.beta_scheduler is not None and not isinstance(self.beta_scheduler, float):
          self.beta_scheduler.step()
        
      if self.lr_scheduler is not None:
        self.lr_coeff = self.lr_scheduler.get_last_lr()[0]
      else:
        self.lr_coeff = self.optimizer.param_groups[0]['lr']

      if isinstance(self.kl_scheduler, float):
        self.kl_coeff = self.kl_scheduler
      elif isinstance(self.kl_scheduler, (LinearScheduler,)):
        self.kl_coeff = self.kl_scheduler.get_last_lr()
      else:
        self.kl_coeff = 0.0

      if isinstance(self.temperature_mean_scheduler, float):
        self.temperature_mean_coeff = self.temperature_mean_scheduler
      elif isinstance(self.temperature_mean_scheduler, (LinearScheduler,)):
        self.temperature_mean_coeff = self.temperature_mean_scheduler.get_last_lr()
      else:
        self.temperature_mean_coeff = 1.0 # default temperature

      if isinstance(self.temperature_individual_scheduler, float):
        self.temperature_individual_coeff = self.temperature_individual_scheduler
      elif isinstance(self.temperature_individual_scheduler, (LinearScheduler,)):
        self.temperature_individual_coeff = self.temperature_individual_scheduler.get_last_lr()
      else:
        self.temperature_individual_coeff = 1.0

      if isinstance(self.lambda_scheduler, float):
        self.lambda_coeff = self.lambda_scheduler
      elif isinstance(self.lambda_scheduler, (LinearScheduler,)):
        self.lambda_coeff = self.lambda_scheduler.get_last_lr()
      else:
        self.lambda_coeff = 0.0 

      if isinstance(self.alpha_scheduler, float):
        self.alpha_coeff = self.alpha_scheduler
      elif isinstance(self.alpha_scheduler, (LinearScheduler,)):
        self.alpha_coeff = self.alpha_scheduler.get_last_lr()
      else:
        self.alpha_coeff = 0.0
      
      if isinstance(self.beta_scheduler, float):
        self.beta_coeff = self.beta_scheduler
      elif isinstance(self.beta_scheduler, (LinearScheduler,)):
        self.beta_coeff = self.beta_scheduler.get_last_lr()
      else:
        self.beta_coeff = 0.0

      if self.writer is not None:
        self.writer.add_scalar('Train/lr_coeff', self.lr_coeff, epoch)
        self.writer.add_scalar('Train/wd_coeff', self.wd_coeff, epoch)
        self.writer.add_scalar('Train/grad_clip_coeff', self.grad_clip_coeff, epoch)
        self.writer.add_scalar('Train/kl_coeff', self.kl_coeff, epoch)
        self.writer.add_scalar('Train/temperature_mean_coeff', self.temperature_mean_coeff, epoch)
        self.writer.add_scalar('Train/temperature_individual_coeff', self.temperature_individual_coeff, epoch)
        self.writer.add_scalar('Train/distance_coeff', self.lambda_coeff, epoch)
        self.writer.add_scalar('Train/alpha', self.alpha_coeff, epoch)
        self.writer.add_scalar('Train/beta', self.beta_coeff, epoch)
        if epoch % self.weight_logging_frequency == 0:
            self.log_weights(epoch)

      if self.verbose:
       logging.info(
            '### Epoch: [%d/%d], Learning rate: %e, Weight decay: %e, Grad clip: %e, KL coeff: %e, Temperature mean coeff: %e, Temperature individual coeff: %e, Distance coeff: %e, Alpha: %e, Beta: %e ###',
                epoch + 1, self.epochs, self.lr_coeff, self.wd_coeff, self.grad_clip_coeff, self.kl_coeff, self.temperature_mean_coeff, self.temperature_individual_coeff, self.lambda_coeff, self.alpha_coeff, self.beta_coeff)

      self.train(train_loader)
      if self.verbose:
        logging.info('#### Train | %s ####', self.train_metrics.get_str())
      self.train_metrics.scalar_logging("Train_%s/" % special_id, epoch)

      if valid_loader is not None:
        self.infer(valid_loader)
        if self.verbose:
          logging.info('#### Valid | %s ####', self.valid_metrics.get_str())
        self.valid_metrics.scalar_logging("Valid_%s/" % special_id, epoch)

    return self.train_time, self.val_time

  def _step(self, input, target, optimizer, metric_container, train_timer):
    start = time.time()
    loss = 0.0
    kl = 0.0 
    distance = 0.0
    wd = 0.0
    teacher_loss_mean = 0.0
    teacher_loss_individual = 0.0
    student_loss = 0.0

    if next(self.model.parameters()).is_cuda:
      input = input.to(next(self.model.parameters()).device)
      target = target.to(next(self.model.parameters()).device)
      
    if optimizer is not None:
      optimizer.zero_grad(set_to_none=True)

    output = self.model(input)

    if optimizer is not None:
      kl = self.model.kl_divergence() if hasattr(self.model, 'kl_divergence') else 0.0
      distance = self.model.weight_distance() if hasattr(self.model, 'weight_distance') else 0.0

      if self.teacher is not None:
        with torch.no_grad():
          teacher_output = self.teacher(input)
        student_loss, teacher_loss_mean, teacher_loss_individual = self.criterion(teacher_output=teacher_output, student_output=output, target=target, temperature_mean= self.temperature_mean_coeff, temperature_individual=self.temperature_individual_coeff)
        loss = (teacher_loss_mean * (1.0-self.beta_coeff) + teacher_loss_individual * self.beta_coeff) * self.alpha_coeff + student_loss * (1.0 - self.alpha_coeff)
      else:
        loss = self.criterion(output=output, target=target)
      wd = self.wd(self.model)
      loss += kl * self.kl_coeff + distance * self.lambda_coeff + wd * self.wd_coeff

      if loss == loss:
        loss.backward()
        if self.grad_clip_coeff > 0:
          self.grad_clip(self.model)
        for p in self.model.parameters():
          if p.grad is not None:
            p.grad[p.grad != p.grad] = 0
        
        optimizer.step()

    metric_container.update(output=output, target=target, loss=loss, teacher_loss_mean=teacher_loss_mean, teacher_loss_individual=teacher_loss_individual, student_loss=student_loss, kl=kl, distance=distance, wd=wd)

    if train_timer:
      self.train_time += time.time() - start
      self.iteration+=1
    else:
      self.val_time += time.time() - start

  def train(self, loader):
    self.train_metrics.reset()
    self.model.train()

    for step, (input, target) in enumerate(loader):
      self._step(
          input=input, target=target, optimizer=self.optimizer, metric_container=self.train_metrics, train_timer=True)
      if step % self.logging_frequency ==0 and self.verbose:
        logging.info('##### Train step: [%03d/%03d] | %s #####',
                     len(loader),  step, self.train_metrics.get_str())
      if self.args is not None and self.args.debug:
        break
      
  @torch.no_grad()
  def infer(self, loader):
    self.valid_metrics.reset()
    self.model.eval()
    
    for step, (input, target) in enumerate(loader):
      self._step(
          input=input, target=target, optimizer=None, metric_container=self.valid_metrics, train_timer=False)

      if step % self.logging_frequency ==0 and self.verbose:
        logging.info('##### Valid step: [%03d/%03d] | %s #####'% 
                      (len(loader), step, self.valid_metrics.get_str()))

      if self.args is not None and self.args.debug:
        break

  def log_weights(self, epoch):
    for name, param in self.model.named_parameters():
      if param.requires_grad:
        self.writer.add_histogram(name.replace('.', '/'), param.data.cpu().numpy(), epoch)
