
import torch.nn as nn
import inspect

class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x, y):
        return x + y

class BasicBlock(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
    ):
        super().__init__()
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1,
                           stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = None
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=False)
        self.add = Add()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.add(out, identity)
        out = self.relu2(out)
        return out

class Classifier(nn.Module):
    def __init__(self, inplanes, output_size, end_layer):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.layers.append(nn.Flatten())
        self.layers.append(end_layer if not inspect.isclass(end_layer) else nn.Linear(inplanes, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Blocks(nn.Module):
    def __init__(self, inplanes, planes, blocks, stride, end_layer=nn.Linear, end=False, output_size=10):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(blocks):
            self.layers.append(BasicBlock(inplanes=inplanes if i == 0 else planes,
                                          planes=planes,
                                          stride=stride if i == 0 else 1))

        self.layers.append(Classifier(planes, output_size, end_layer) if end else nn.Identity())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ResNet(nn.Module):
    def __init__(
        self,
        layers,
        planes=[32, 64, 128, 256],
        strides=[1, 2, 2, 2],
        output_size=10,
        end_layer=nn.Linear,
        end=True
    ):
        super().__init__()
        self.output_size = output_size

        assert len(planes) == len(layers)

        self.layers = nn.ModuleList(
            [nn.Conv2d(3, planes[0], kernel_size=3,
                                stride=1, padding=1, bias=False),
             nn.BatchNorm2d(planes[0]),
                nn.ReLU(inplace=True)])
        
        for i in range(len(layers)):
            self.layers.append(Blocks(
                inplanes=planes[0] if i == 0 else planes[i-1],
                planes=planes[i],
                blocks=layers[i],
                stride=strides[i],
                end=end and i == len(layers) - 1,
                end_layer=end_layer,
                output_size=output_size
            ))

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x