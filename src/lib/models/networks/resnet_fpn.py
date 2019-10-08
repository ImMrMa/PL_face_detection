import os
import math
import torch
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F
__all__ = [
    'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
]

# you need to download the models to ~/.torch/models
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }
models_dir = os.path.expanduser('~/.torch/models')
model_name = {
    'resnet18': 'resnet18-5c106cde.pth',
    'resnet34': 'resnet34-333f7ec4.pth',
    'resnet50': 'resnet50-19c8e357.pth',
    'resnet101': 'resnet101-5d3b4d8f.pth',
    'resnet152': 'resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        if stride != 1:
            self.conv1 = nn.Conv2d(inplanes,
                                   planes,
                                   kernel_size=4,
                                   stride=stride,
                                   padding=1,
                                   bias=False)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNetMa(nn.Module):
    def __init__(self, block, layers, num_classes=1000, fpn=False):
        super(ResNetMa, self).__init__()
        self.fpn = fpn
        self.inplanes = 32
        self.conv1 = nn.Conv2d(3,
                               32,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 32, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 512, layers[4], stride=2)
        self.layer6 = self._make_layer(block, 1024, layers[5], stride=2)
        self.fpn_net = FPN()
        # self.layer7 = self._make_layer(block, 1024, layers[6], stride=2)
        # self.layer8 = self._make_layer(block, 512, layers[7], stride=2)
        if not fpn:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride != 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes,
                              planes * block.expansion,
                              kernel_size=4,
                              stride=stride,
                              padding=1,
                              bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes,
                              planes * block.expansion,
                              kernel_size=1,
                              stride=stride,
                              bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        layers_p = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x1 = self.layer3(x)
        layers_p.append(x1)
        x2 = self.layer4(x1)
        layers_p.append(x2)
        x3 = self.layer5(x2)
        layers_p.append(x3)
        x4 = self.layer6(x3)
        layers_p.append(x4)
        if self.fpn:
            p = self.fpn_net(layers_p)
            return p
        else:
            x = self.avgpool(x4)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x


class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        self.top_layer = nn.Conv2d(1024,
                                   256,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)

        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # Lateral layers
        self.latlayer1 = nn.Conv2d(128,
                                   256,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)
        self.latlayer2 = nn.Conv2d(256,
                                   256,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)
        self.latlayer3 = nn.Conv2d(512,
                                   256,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)

    def upsample_add(self, x, y):
        _, _, H, W = y.shape
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, layers):
        p4 = self.top_layer(layers[3])
        layer3 = self.latlayer3(layers[2])
        p3 = self.upsample_add(p4, layer3)
        layer2 = self.latlayer2(layers[1])
        p2 = self.upsample_add(p3, layer2)
        layer1 = self.latlayer1(layers[0])
        p1 = self.upsample_add(p2, layer1)
        return p1


def resnet_ma_32_obj(pretrained=False, **kwargs):
    model = ResNetMa(BasicBlock, [1, 1, 2, 2, 2, 2, 2, 2], **kwargs)
    # if pretrained:
    #     model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet18'])))
    return model


def resnet_ma_32_fpn(pretrained=True, fpn=True, **kwargs):
    model = ResNetMa(BasicBlock, [1, 1, 2, 2, 2, 2, 2, 2], fpn=fpn, **kwargs)
    if pretrained:
        params = torch.load(
            '/home/mayx/project/resnet_new_loader/log/resnet_obj-07_20-11_04/86_0.81-19_56.pth'
        )
        params_clean = dict()
        for k in list(params['model'].keys()):
            if 'fc' not in k:
                params_clean[k[7:]] = params['model'][k]
        params_model = model.state_dict()
        params_model.update(params_clean)
        model.load_state_dict(params_model)
    return model


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class ResnetMaDect(nn.Module):
    def __init__(self, num_layers, heads, head_conv, pretrained=True):
        super(ResnetMaDect, self).__init__()
        model = resnet_ma_32_fpn(fpn=True, pretrained=pretrained)
        self.backbone = model
        self.heads = heads
        self.head_conv = head_conv
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(256,
                              head_conv,
                              kernel_size=3,
                              padding=1,
                              bias=True), nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv,
                              classes,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(256,
                               classes,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.backbone(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]


def get_resnet_ma(num_layers, heads, head_conv=256):
    deeplab_model = ResnetMaDect(num_layers, heads, head_conv)
    return deeplab_model
