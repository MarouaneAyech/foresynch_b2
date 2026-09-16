"""IResNet-50, conforme à arcface_torch. Repris tel quel du notebook d'origine."""
import torch
import torch.nn as nn
from torch.nn import BatchNorm1d, BatchNorm2d, Conv2d, Dropout, Linear, PReLU, Sequential


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super().__init__()
        self.bn1 = BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = Conv2d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn2 = BatchNorm2d(planes, eps=1e-05)
        self.prelu = PReLU(planes)
        self.conv2 = Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn3 = BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Module):
    fc_scale = 7 * 7

    def __init__(self, block, layers, dropout=0, num_features=512, fp16=False):
        super().__init__()
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        self.conv1 = Conv2d(3, self.inplanes, 3, 1, 1, bias=False)
        self.bn1 = BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn2 = BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = Dropout(p=dropout, inplace=True)
        self.fc = Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                BatchNorm2d(planes * block.expansion, eps=1e-05),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return Sequential(*layers)

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)
        return x


def iresnet50(**kwargs) -> IResNet:
    return IResNet(IBasicBlock, [3, 4, 14, 3], **kwargs)
