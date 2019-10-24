'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        if stride!=1:
            self.conv1=nn.Sequential(
            nn.Conv2d(in_planes, self.expansion*planes, kernel_size=4, stride=stride,padding=1, bias=False),
            nn.BatchNorm2d(self.expansion*planes)
        )
        else:
            self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, self.expansion*planes, kernel_size=3, stride=stride,padding=1, bias=False),
            nn.BatchNorm2d(self.expansion*planes)
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if stride!=1:
                self.shortcut=nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=2, stride=stride,padding=0, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
            else:
                self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class L2Norm(nn.Module):
    def __init__(self, inplanes, gamma_init=10):
        super(L2Norm, self).__init__()
        self.gamma_init = torch.Tensor(1, inplanes, 1, 1)
        self.gamma_init[...] = gamma_init
        self.gamma_init = Parameter(self.gamma_init)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = x * self.gamma_init
        return x
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512, num_classes)
        self.norm1=L2Norm(16)
        self.norm2=L2Norm(16)
        self.norm3=L2Norm(16)
        self.norm4=L2Norm(16)
        self.trans1=nn.Conv2d(64,
                     16,
                     kernel_size=1,
                     stride=1,
                     padding=0,
                     bias=False)
        self.trans2=nn.ConvTranspose2d(in_channels=128,
                                        out_channels=16,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1,
                                        output_padding=0,
                                        bias=False)
        self.trans3=nn.ConvTranspose2d(in_channels=256,
                                        out_channels=16,
                                        kernel_size=4,
                                        stride=4,
                                        padding=0,
                                        output_padding=0,
                                        bias=False)
        self.trans4=nn.ConvTranspose2d(512,
                     16,
                     kernel_size=4,
                     stride=4,
                     padding=0,
                     bias=False)
        self.s_conv=nn.Conv2d(64,
                     16,
                     kernel_size=3,
                     stride=1,
                     padding=1,
                     bias=False)
        self.hm=nn.Sequential( nn.Conv2d(16,
             1,
             kernel_size=1,
             stride=1,
             padding=0,
             bias=True),nn.Sigmoid())
        self.wh=nn.Conv2d(16,
             2,
             kernel_size=1,
             stride=1,
             padding=0,
             bias=True)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        s1 = self.layer1(x)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        s1=self.trans1(s1)
        s1=self.norm1(s1)
        s2=self.trans2(s2)
        s2=self.norm2(s2)
        s3=self.trans3(s3)
        s3=self.norm3(s3)
        s4=self.trans4(s4)
        s4=F.interpolate(s4,
                        scale_factor=2,
                        mode='bilinear')
        s4=self.norm4(s4)
        s=torch.cat([s1,s2,s3,s4],1)
        s=self.s_conv(s)
        hm_small=self.hm(s)
        wh_small=self.wh(s)
        return dict(hm_small=hm_small,wh_small=wh_small)

def ResNet18(**kwargs):
    model=ResNet(BasicBlock, [2,2,2,2,2],**kwargs)
    state_dict=torch.load('/data/users/mayx/project/github/CenterNet/models/resnet18_64_conv4_conv2.pth',map_location='cpu')
    state_dict=state_dict['net']
    model_dict=model.state_dict()
    # for k, v in model.named_parameters():
    #     if 'hm' in k:
    #         if 'bias' in k:
    #             v = torch.ones_like(v) * -math.log(
    #                 (1 - 0.01) / 0.01)
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    for k,v in model.named_parameters():
        v.requires_grad=False
        if 'layer4' in k:
            break
    for k,v in model.named_parameters():
        if 'bn' in k:
            v.requires_grad=False
    for k,v in model.named_parameters():
        print(k,v.requires_grad)
    input('s')
    return ResNet(BasicBlock, [2,2,2,2,2],**kwargs)

def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3,4,6,3],**kwargs)

def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3,4,6,3],**kwargs)

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
