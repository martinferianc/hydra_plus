import numpy as np
import torch.nn as nn

class LinearScheduler():
    def __init__(self, start_epoch, end_epoch, epochs, start_value = 0.0, end_value=1.0):
        assert end_epoch >= start_epoch
        assert end_value >= 0
        self.vals = [0.] * start_epoch + list(np.linspace(
            start_value, end_value, end_epoch-start_epoch)) + [end_value]*(epochs - end_epoch)
        self.current_index = 0

        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.epochs = epochs
        self.start_value = start_value
        self.end_value = end_value
      
    def __getitem__(self, index):
        return self.vals[index]
        
    def step(self):
        self.current_index+=1
        
    def get_last_lr(self):
        return self.vals[self.current_index]
    
    def reset(self):
        self.current_index = 0

    def __repr__(self):
        return "LinearScheduler(start_epoch=%d, end_epoch=%d, epochs=%d, start_value=%f, end_value=%f)" % (self.start_epoch, self.end_epoch, self.epochs, self.start_value, self.end_value)


class WeightDecay():
    def __init__(self, coeff=0.0):
        self.coeff = coeff
        
    def l2(self, p):
        return 0.5 * p.pow(2).sum()

    def __call__(self, model):
        l2_loss = 0
        for _, p in model.named_parameters():
            if p.requires_grad is False:
                continue
            # Check that this is not Gauss
            if hasattr(p, '_no_wd'):
                continue
            l2_loss += self.l2(p)
        return l2_loss

class GradientClipping():
    def __init__(self, coeff=5.0):
        self.coeff = coeff

    def __call__(self, model):
        params = []
        for _, p in model.named_parameters():
            if p.requires_grad is False:
                continue
            # Check that none of the exclude_phrases is in the name
            if hasattr(p, '_no_grad_clip'):
                continue
            params.append(p)
        nn.utils.clip_grad_norm_(params, self.coeff)