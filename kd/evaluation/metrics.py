import torch
import torch.nn.functional as F

METRICS_MAPPING = {
    "error": "Error [%]",
    "ece": "Expected Calibration Error [%]",
    "entropy": "Entropy [nats]",
    "rmse": "Root Mean Squared Error",
    "mae": "Mean Absolute Error",
    "nll": "Negative LL [nats]",
    "loss": "Total Loss [nats]",
    "teacher_loss_mean": "Teacher Loss Mean [nats]",
    "teacher_loss_individual": "Teacher Loss Individual [nats]",
    "student_loss": "Student Loss [nats]",
    "f1": "F1 Score [0-1]",
    "kl": "KL Divergence [nats]",
    "wd": "Weight Decay",
    "distance": "Distance",
    "flops": "FLOPs",
    "model_size": "Model Size [MB]",
    "memory_footprint": "Memory Footprint [MB]",
    "params": "Total Parameters",
}

METRICS_DESIRED_TENDENCY_MAPPING = {
    "error": "down",
    "ece": "down",
    "entropy": "up",
    "rmse": "down",
    "mae": "down",
    "nll": "down",
    "loss": "down",
    "teacher_loss_mean": "down",
    "teacher_loss_individual": "down",
    "student_loss": "down",
    "f1": "up",
    "kl": "down",
    "wd": "down",
    "distance": "down",
    "flops": "down",
    "model_size": "down",
    "memory_footprint": "down",
    "params": "down"
}

ALGORITHMIC_METRICS = ["error", "ece", "entropy", "rmse", "mae", "nll", "f1"]
HARDWARE_METRICS= ["flops", "model_size", "memory_footprint", "params"]

class Metric():
    def __init__(self, writer=None, output_size=None):
        self.writer = writer 
        self.metrics = []
        self.metric_labels = []
        self.output_size = output_size

        self.loss = AverageMeter()
        self.teacher_loss_mean = AverageMeter()
        self.teacher_loss_individual = AverageMeter()
        self.student_loss = AverageMeter()
        self.kl = AverageMeter()
        self.distance = AverageMeter()
        self.wd = AverageMeter()

        self.metrics = [self.loss, self.teacher_loss_mean, self.teacher_loss_individual, self.student_loss, self.kl, self.distance, self.wd]
        self.metric_labels = ["loss", "teacher_loss_mean", "teacher_loss_individual", "student_loss", "kl", "distance", "wd"]

    def reset(self):
        for metric in self.metrics:
            if hasattr(metric, "reset"):
                metric.reset()

    def scalar_logging(self, info, iteration):
        if self.writer is None:
            return
        for i, metric in enumerate(self.metrics):
            val = metric.avg if hasattr(metric, 'avg') else metric()
            self.writer.add_scalar(info+METRICS_MAPPING[self.metric_labels[i]], val, iteration)

    def get_str(self):
        s = ""
        for i, metric in enumerate(self.metrics):
            val = metric.avg if hasattr(metric, 'avg') else metric()
            s += f'{METRICS_MAPPING[self.metric_labels[i]]}: {str(val)} '
        return s

    def get_packed(self):
        d = {}
        for i, metric in enumerate(self.metrics):
            val = metric.avg if hasattr(metric, 'avg') else metric()
            d[self.metric_labels[i].lower()] =  val
        return d

    def update(self, loss=0.0, teacher_loss_mean = 0.0, teacher_loss_individual = 0.0, student_loss = 0.0, kl = 0.0, distance = 0.0, wd=0.0):
        loss = loss if isinstance(loss, float) else loss.item()
        teacher_loss_mean = teacher_loss_mean if isinstance(teacher_loss_mean, float) else teacher_loss_mean.item()
        teacher_loss_individual = teacher_loss_individual if isinstance(teacher_loss_individual, float) else teacher_loss_individual.item()
        student_loss = student_loss if isinstance(student_loss, float) else student_loss.item()
        kl = kl if isinstance(kl, float) else kl.item()
        distance = distance if isinstance(distance, float) else distance.item()
        wd = wd if isinstance(wd, float) else wd.item()

        self.loss.update(loss,1)
        self.teacher_loss_mean.update(teacher_loss_mean,1)
        self.teacher_loss_individual.update(teacher_loss_individual,1)
        self.student_loss.update(student_loss,1)
        self.kl.update(kl,1)
        self.distance.update(distance,1)
        self.wd.update(wd,1)

    def get_key_metric(self):
        pass

    def get_key_metric_label(self):
        pass
        
class ClassificationMetric(Metric):
    def __init__(self, writer, output_size):
        super(ClassificationMetric, self).__init__(writer, output_size)
        self.confusion_matrix = torch.zeros((self.output_size, self.output_size))
        
        self.entropy = AverageMeter()
        self.ece = AverageMeter()
        self.nll = AverageMeter()

        self.metrics += [self.nll, self.error, self.entropy, self.ece, self.f1]
        self.metric_labels += ["nll", "error", "entropy", "ece", "f1"]

    @staticmethod
    def _get_predictions(output):
        bs, _ = output.size()
        tensor = output.data
        _, indices = tensor.max(1)
        indices = indices.view(bs, 1)
        return indices

    @staticmethod
    def _ec_error(output, target):
        _ece = 0.0
        confidences, predictions = torch.max(output, 1)
        accuracies = predictions.eq(target)

        bin_boundaries = torch.linspace(0, 1, 10 + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * \
                confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                _ece += torch.abs(avg_confidence_in_bin -
                                    accuracy_in_bin) * prop_in_bin
        _ece = _ece if isinstance(_ece, float) else _ece.item()
        return _ece

    @torch.no_grad()
    def update(self, output, target, loss=0.0, teacher_loss_mean = 0.0, teacher_loss_individual = 0.0, student_loss = 0.0, kl = 0.0, distance = 0.0, wd=0.0):
        super(ClassificationMetric, self).update(loss=loss, teacher_loss_mean=teacher_loss_mean, teacher_loss_individual=teacher_loss_individual, student_loss=student_loss, kl=kl, distance=distance, wd=wd)
        if len(output.shape) == 2:
            output = output.unsqueeze(1)
        output = output.detach()
        output = F.softmax(output, dim=2)
        output = output.mean(dim=1)
        bs, _ = output.size()
        predictions = ClassificationMetric._get_predictions(output)
        for lt, lp in zip(target, predictions):
            if target.is_cuda or predictions.is_cuda and not self.confusion_matrix.is_cuda:
                self.confusion_matrix = self.confusion_matrix.to(predictions.device)
            self.confusion_matrix += self._fast_hist(lt.view(-1), lp.view(-1))
        nll = F.nll_loss(torch.log(output+1e-8), target.long(), reduction='mean').item()
        ece =  ClassificationMetric._ec_error(output.view(-1, self.output_size), target.view(-1))*100
        entropy = -(torch.sum(torch.log(output+1e-8)*output)/bs).item()

        self.nll.update(nll, bs)
        self.ece.update(ece, bs)
        self.entropy.update(entropy, bs)
    
    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.output_size)
        return torch.bincount(
            self.output_size * label_true[mask].long() + label_pred[mask],
            minlength=self.output_size ** 2,
        ).reshape(self.output_size, self.output_size)

    def error(self):
        hist = self.confusion_matrix
        true_pos = torch.diag(hist) 
        # The eta is added to avoid nans 
        return (1.-(true_pos.sum() / (hist.sum()+1e-8)).item())*100
    
    def f1(self):
        hist = self.confusion_matrix
        if self.output_size == 2: # this is simply one-vs-one classification
            precision = (torch.diag(hist) / (hist.sum(1) + 1e-8))[0]
            recall = (torch.diag(hist) / (hist.sum(0) + 1e-8))[0]
        else:
            precision = (torch.diag(hist) / (hist.sum(1) + 1e-8)).mean()
            recall = (torch.diag(hist) / (hist.sum(0) + 1e-8)).mean()
        return (2*(precision*recall)/(precision+recall + 1e-8)).item()
            
    def reset(self):
        super().reset()
        self.confusion_matrix = torch.zeros((self.output_size, self.output_size))

    def get_key_metric(self):
        return self.error()

    def get_key_metric_label(self):
        return "ERROR"
    
class RegressionMetric(Metric):
    def __init__(self, writer):
        super(RegressionMetric, self).__init__(writer, None)
                
        self.rmse = AverageMeter()
        self.nll = AverageMeter()
        self.mae = AverageMeter()

        self.metrics += [self.rmse, self.nll, self.mae]
        self.metric_labels += ["rmse", "nll", "mae"]

    @torch.no_grad()
    def update(self, output, target, loss=0.0, teacher_loss_mean = 0.0, teacher_loss_individual = 0.0, student_loss = 0.0, kl = 0.0, distance = 0.0, wd=0.0):
        super(RegressionMetric, self).update(loss=loss, teacher_loss_mean=teacher_loss_mean, teacher_loss_individual=teacher_loss_individual, student_loss=student_loss, kl=kl, distance=distance, wd=wd)
        if len(output.shape) == 2:
            output = output.unsqueeze(1)
        output = output.detach()
        mean, var = output[:,:,0], output[:,:,1].exp()
        mean_var = mean.var(dim=1)
        # Replace nans with zeros, this is if the sample size is 1
        mean_var[torch.isnan(mean_var)] = 0.0
        var = mean_var + var.mean(dim=1)
        mean = mean.mean(dim=1)
        bs = mean.size(0)
        target = target.squeeze(1)
        
        rmse = torch.sqrt(torch.mean((mean-target)**2)+1e-8).item()
        nll = F.gaussian_nll_loss(mean, target, var.clamp(min=1e-8), reduction='mean').item()
        mae = torch.mean(torch.abs(mean-target)).item()

        self.rmse.update(rmse, bs)
        self.nll.update(nll, bs)
        self.mae.update(mae, bs)     

    def get_key_metric(self):
        return self.nll.avg

    def get_key_metric_label(self):
        return "NLL"
    
class AverageMeter(object):
      def __init__(self):
        self.reset()

      def reset(self):
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

      def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt