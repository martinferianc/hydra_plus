import torch.nn as nn
import torch

class WeightDistance(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, layers):
        distance = 0.0
        
        # Calculate the mean weight across all layers
        mean_weight = torch.mean(torch.stack([layer.weight for layer in layers]), dim=0)
        if layers[0].bias is not None:
            mean_bias = torch.mean(torch.stack([layer.bias for layer in layers]), dim=0)

        single_distance = 0.0
        for i in range(len(layers)):
            if len(mean_weight.shape) == 1:
                single_distance = self.cos(mean_weight.reshape(-1, 1), layers[i].weight.reshape(-1, 1))
            elif len(mean_weight.shape) == 2:
                single_distance = self.cos(mean_weight, layers[i].weight)
            elif len(mean_weight.shape) == 4:
                single_distance = self.cos(mean_weight.reshape(mean_weight.shape[0], -1), layers[i].weight.reshape(layers[i].weight.shape[0], -1)) 
            
            if layers[i].bias is not None:
                single_distance += self.cos(mean_bias.reshape(-1, 1), layers[i].bias.reshape(-1, 1))
            distance += single_distance
        return distance 

    def cos(self, weight1, weight2):
        nominator = (weight1*weight2).sum(dim=1)
        denominator = (torch.sqrt(weight1.pow(2).sum(dim=1)+1e-8)*torch.sqrt(weight2.pow(2).sum(dim=1)+1e-8))+1e-8
        return ((nominator/denominator+1.0)/2.0).sum()

"""
class WeightDistance(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, layers):
        distance = 0.0
        
        # Calculate the mean weight across all layers
        weight = torch.stack([layer.weight for layer in layers])
        weight = weight.reshape(weight.shape[0], -1)
        if layers[0].bias is not None:
            bias = torch.stack([layer.bias for layer in layers])
            bias = bias.reshape(bias.shape[0], -1)

        distance=0.0
        distance += torch.log(torch.linalg.det(torch.matmul(weight, weight.t()))+1e-8)
        
        if layers[0].bias is not None:
            distance += torch.log(torch.linalg.det(torch.matmul(bias, bias.t()))+1e-8)
        return distance 
# This was the reimplementation of the "log det" distance, 
# which however empirically was not working well if tried on the spiral or regression datasets.
"""