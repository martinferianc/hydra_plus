# Define simple linear model with ReLU activation
import torch.nn as nn
import numpy as np
import inspect

class FC(nn.Module):
    def __init__(self, input_size, output_size, planes, end_layer=nn.Linear, activation_end=False):
        super(FC, self).__init__()
        self.input_size = input_size
        if isinstance(self.input_size, (tuple, list)):
            self.input_size = np.prod(self.input_size)
            
        self.output_size = output_size
        self.layers = nn.ModuleList([nn.Flatten()])

        for i in range(len(planes)+1):
            if i == 0 and i == len(planes):
                self.layers.append(end_layer if not inspect.isclass(end_layer) else nn.Linear(self.input_size, self.output_size))
            elif i == 0:
                self.layers.append(nn.Linear(self.input_size, planes[0]))
                self.layers.append(nn.ReLU())
            elif i == len(planes):
                self.layers.append(end_layer if not inspect.isclass(end_layer) else nn.Linear(planes[-1], self.output_size))
            else:
                self.layers.append(nn.Linear(planes[i-1], planes[i]))
                self.layers.append(nn.ReLU())

        if activation_end:
            self.layers.append(nn.ReLU())

        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, input):
        return self.layers(input)
        


        
        
        