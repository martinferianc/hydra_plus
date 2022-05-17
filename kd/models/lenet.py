import torch.nn as nn
import inspect

class Classifier(nn.Module):
    def __init__(self, planes, end_layer=nn.Linear):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(1, len(planes)):
            if i == len(planes) - 1:
                self.layers.append(
                    nn.Linear(planes[i - 1], planes[i])
                    if inspect.isclass(end_layer)
                    else end_layer
                )

            else:
                self.layers.append(nn.Linear(planes[i-1], planes[i]))
            if i!=len(planes)-1:
                self.layers.append(nn.ReLU(inplace=True))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class LeNetFeatureExtractor(nn.Module):
    def __init__(self, planes, output_size, end=False, end_layer=nn.Linear, end_planes=[]): 
        super(LeNetFeatureExtractor, self).__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Conv2d(3, planes[0], kernel_size=3,
                                        stride=1, padding=1, bias=True))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        for i in range(1, len(planes)):
            self.layers.append(nn.Conv2d(planes[i-1], planes[i], kernel_size=3,
                                            stride=1, padding=1, bias=True))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layers.append(nn.AdaptiveAvgPool2d((4, 4)))
        self.layers.append(nn.Flatten())
        if end:
            self.layers.append(Classifier(planes=end_planes, end_layer=end_layer))
        self.output_size = output_size

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x