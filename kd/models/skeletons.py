import torch.nn as nn
import torch

from kd.models.distance import WeightDistance
from kd.models.resnet import BasicBlock, Classifier
# Define a multi-tail runner with respect to a tail model
# That is being deepcopied
class MultiTails(nn.Module):
    def __init__(self, tails, n_tails, n_predictions, method=""):
        super().__init__()
        # n_tails and n_predictions can be different e.g. one tail can be used multiple times
        # Each tail should have all layers in .layers for easy access
        self.n_tails = n_tails
        self.n_predictions = n_predictions
        assert self.n_predictions>=self.n_tails

        self.tails = nn.ModuleList(tails)

        self.method = method
        # Define distance function
        self.distance =  None
        if method == "hydra+":
            self.distance = WeightDistance()

    # Forward features from the shared core
    def forward(self, features):
        features = features.unsqueeze(0).expand(self.n_predictions, *features.shape).contiguous().clone()
        if self.method in ["endd", "drop", "gauss"]:
            # There is actually just one tail but it is being sampled at the end
            tails = [0] * self.n_predictions
        elif self.method in ["ensemble", "hydra", "hydra+"]:
            # Make sure to utilise all the tails, in the same order
            tails = list(range(self.n_tails))
        else:
            raise ValueError("Unknown method: {}".format(self.method))

        outputs = []
        for i, tail in enumerate(tails):
            output = self.tails[tail](features[i,:])
            outputs.append(output)
        return torch.stack(outputs, dim=1)

    def weight_distance(self):
        if self.distance is None:
            return 0.0
        distance = 0.0
        for i in range(len(self.tails[0].layers)):
            if isinstance(self.tails[0].layers[i], (nn.Linear, nn.Conv2d)):
                distance += self.distance([self.tails[k].layers[i] for k in range(len(self.tails))])
            elif isinstance(self.tails[0].layers[i], BasicBlock):
                distance += self.distance([self.tails[k].layers[i].conv1 for k in range(len(self.tails))])
                distance += self.distance([self.tails[k].layers[i].conv2 for k in range(len(self.tails))])
                if self.tails[0].layers[i].downsample is not None:
                    distance += self.distance([self.tails[k].layers[i].downsample[0] for k in range(len(self.tails))])
            elif isinstance(self.tails[0].layers[i], Classifier):
                distance += self.distance([self.tails[k].layers[i].layers[-1] for k in range(len(self.tails))])
        return distance

class BaseModel(nn.Module):
    def __init__(self, core_model=nn.Identity(), tail_model=nn.Identity(), output_size=-1):
        super().__init__()
        self.core_model = core_model
        self.tail_model = tail_model
        self.output_size = output_size

    def forward(self, x):
        x = self.core_model(x)
        x = self.tail_model(x)
        return x

    def kl_divergence(self):
        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_divergence') and not isinstance(module, BaseModel):
                kl += module.kl_divergence()
        return kl

    def weight_distance(self):
        return self.tail_model.weight_distance()