import torch.nn as nn
from torch.nn.parameter import Parameter
from torchvision.models.utils import load_state_dict_from_url
import torch.nn.functional as F
import torch
import os.path as osp
import math
__all__ = [
    'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d'
]

model_urls = {
    'resnet18':
    'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34':
    'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50':
    'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101':
    'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152':
    'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d':
    'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d':
    'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}


def group_norm(in_planes):
    return nn.GroupNorm(in_planes // 16, in_planes)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv4x4(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=4,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv2x2(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=2,
                     stride=stride,
                     padding=0,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None,
                 conv4=False,
                 conv2=False,
                 replace_with_bn=False):
        super(BasicBlock, self).__init__()
        self.norm_layer = norm_layer
        if conv4 and conv2:
            raise ValueError('wrong!')
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        # if dilation > 1:
        #     raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if conv4 and dilation == 1:
            self.conv1 = conv4x4(inplanes, planes, stride, dilation=dilation)
        elif conv2 and dilation == 1:
            self.conv1 = conv2x2(inplanes, planes, stride, dilation=dilation)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        if replace_with_bn:
            norm_layer = group_norm
        self.bn1 = norm_layer(planes)
        norm_layer = self.norm_layer
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None,
                 conv4=False,
                 conv2=False,
                 replace_with_bn=False):
        super(Bottleneck, self).__init__()
        self.norm_layer = norm_layer
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        if conv4 and dilation == 1:
            self.conv2 = conv4x4(width, width, stride, groups)
        elif conv2 and dilation == 1:
            self.conv2 = conv2x2(width, width, stride, groups)
        else:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        if replace_with_bn:
            norm_layer = group_norm
        self.bn2 = norm_layer(width)
        norm_layer = self.norm_layer
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None,
                 conv4=False,
                 conv2=False,
                 change_s1=False,
                 conv4_conv2=False,
                 replace_with_bn=False,
                 all_gn=False):
        super(ResNet, self).__init__()
        if all_gn:
            bn_layer = group_norm
        else:
            bn_layer = nn.BatchNorm2d
        print(all_gn)
        print(bn_layer)
        input('s')
        norm_layer = bn_layer
        self._norm_layer = norm_layer

        if block is BasicBlock:
            transform_planes = 128
        elif block is Bottleneck:
            transform_planes = 256
        else:
            raise ValueError('not block')
        self.inplanes = 64
        self.dilation = 1
        self.change_s1 = change_s1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(
                                 replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        if change_s1:
            if replace_with_bn:
                norm_layer = group_norm
            self.inplanes = 16
            
            self.layeri0 = nn.Sequential(
                nn.Conv2d(3,
                          16,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False), norm_layer(self.inplanes),
                nn.ReLU(inplace=True))
            self.layeri1 = self._make_layer(block,
                                            16,
                                            1,
                                            stride=1,
                                            conv4_conv2=conv4_conv2)
            self.layer0 = self._make_layer(block,
                                           32,
                                           2,
                                           stride=2,
                                           conv2=conv2,
                                           conv4=conv4,
                                           conv4_conv2=conv4_conv2)
            norm_layer = self._norm_layer
            self.layer1 = self._make_layer(block,
                                           64,
                                           layers[0],
                                           stride=2,
                                           conv2=conv2,
                                           conv4=conv4,
                                           conv4_conv2=conv4_conv2)
        else:
            self.conv1 = nn.Conv2d(3,
                                   self.inplanes,
                                   kernel_size=7,
                                   stride=2,
                                   padding=3,
                                   bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block,
                                           64,
                                           layers[0],
                                           stride=1,
                                           conv4_conv2=conv4_conv2)
        self.inplanes_s2 = self.inplanes
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       conv4=conv4,
                                       conv2=conv2,
                                       conv4_conv2=conv4_conv2)
        self.inplanes_s3 = self.inplanes
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       conv4=conv4,
                                       conv2=conv2,
                                       conv4_conv2=conv4_conv2)
        self.inplanes_s4 = self.inplanes
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       conv4_conv2=conv4_conv2)
        self.inplanes_s5 = self.inplanes
        self.s3_up = nn.ConvTranspose2d(in_channels=self.inplanes_s3,
                                        out_channels=transform_planes,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1,
                                        output_padding=0,
                                        bias=False)
        self.s4_up = nn.ConvTranspose2d(in_channels=self.inplanes_s4,
                                        out_channels=transform_planes,
                                        kernel_size=4,
                                        stride=4,
                                        padding=0,
                                        output_padding=0,
                                        bias=False)
        self.s5_up = nn.ConvTranspose2d(in_channels=self.inplanes_s5,
                                        out_channels=transform_planes,
                                        kernel_size=4,
                                        stride=4,
                                        padding=0,
                                        output_padding=0,
                                        bias=False)
        self.s1_up = nn.ConvTranspose2d(in_channels=32,
                                        out_channels=32,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1,
                                        output_padding=0,
                                        bias=False)
        self.s2_up = nn.ConvTranspose2d(in_channels=64,
                                        out_channels=32,
                                        kernel_size=4,
                                        stride=4,
                                        padding=0,
                                        output_padding=0,
                                        bias=False)
        self.s3_transform= nn.Conv2d(in_channels=256,out_channels=64,batch_size=1,stride=1,padding=0,bias=False)
        self.n_img=L2Norm(3)
        self.n0=L2Norm(16)
        self.n1=L2Norm(32)
        self.n2=L2Norm(32)
        self.n3 = L2Norm(transform_planes)
        self.n4 = L2Norm(transform_planes)
        self.n5 = L2Norm(transform_planes)
        self.s_small_conv=nn.Sequential(
            nn.Conv2d(in_channels=3+16+32+32+32,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), norm_layer(64),
            nn.ReLU(inplace=True))
        self.hm_small=nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0,
            ), nn.Sigmoid())
        self.wh_small = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=2,
                kernel_size=1,
                stride=1,
                padding=0,
            ))
        self.s_conv = nn.Sequential(
            nn.Conv2d(in_channels=transform_planes * 3,
                      out_channels=transform_planes,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), norm_layer(transform_planes),
            nn.ReLU(inplace=True))
        self.hm = nn.Sequential(
            nn.Conv2d(
                in_channels=transform_planes,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0,
            ), nn.Sigmoid())
        self.wh = nn.Sequential(
            nn.Conv2d(
                in_channels=transform_planes,
                out_channels=2,
                kernel_size=1,
                stride=1,
                padding=0,
            ))
        self.offset = nn.Sequential(
            nn.Conv2d(
                in_channels=transform_planes,
                out_channels=2,
                kernel_size=1,
                stride=1,
                padding=0,
            ))
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1,
                    dilate=False,
                    conv4=False,
                    conv2=False,
                    conv4_conv2=False,
                    replace_with_bn=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if conv4_conv2 and stride != 1:
                if replace_with_bn:
                    norm_layer = group_norm
                downsample = nn.Sequential(
                    conv2x2(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
                norm_layer = self._norm_layer
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

        layers = []
        layers.append(
            block(self.inplanes,
                  planes,
                  stride,
                  downsample,
                  self.groups,
                  self.base_width,
                  previous_dilation,
                  norm_layer,
                  conv4=conv4,
                  conv2=conv2,
                  replace_with_bn=replace_with_bn))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,
                      norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        s_img=self.n_img(x)
        if self.change_s1:
            x = self.layeri0(x)
            s0 = self.layeri1(x)
            s1 = self.layer0(s0)
            s2 = self.layer1(s1)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            s2 = self.layer1(x)

        s3 = self.layer2(s2)
        s4 = self.layer3(s3)
        s5 = self.layer4(s4)
        s1 = self.s1_up(s1)
        s2 = self.s2_up(s2)
        s3 = self.s3_up(s3)
        s4 = self.s4_up(s4)
        s5 = self.s5_up(s5)
        s3_small=self.s3_transform(s3)
        s3_small = F.interpolate(s3_small,scale_factor=4, mode='bilinear')
        s0=self.n0(s0)
        s1=self.n1(s1)
        s2=self.n2(s2)
        s_small_cat=torch.cat([s_img,s0,s1,s2,s3_samll],1)
        s_small_cat=self.s_small_conv(s_small_cat)
        hm_small=self.hm_small(s_small_cat)
        wh_small=self.wh_small(s_small_cat)
        s3 = self.n3(s3)
        s4 = self.n4(s4)
        s5 = self.n5(s5)
        s_cat = torch.cat([s3, s4, s5], 1)
        s_cat = self.s_conv(s_cat)
        hm = self.hm(s_cat)
        wh = self.wh(s_cat)
        offset = self.offset(s_cat)
        return dict(hm=hm, wh=wh, offset=offset,hm_small=hm_small,wh_small=wh_small)

    def init_weights(
            self,
            pretrained='',
    ):
        if osp.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            if 'state_dict' in pretrained_dict.keys():
                pretrained_dict = pretrained_dict['state_dict']
        elif pretrained:
            pretrained_dict = load_state_dict_from_url(model_urls[pretrained])
            # self.load_state_dict(pretrained_dict)
        model_dict = self.state_dict()
        if pretrained:
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict.keys() and v.shape == model_dict[k].shape and 'bn' not in k
            }

            for k,v in self.named_parameters():
                v.requires_grad=False
                if 'layer2' in k:
                    break
        else:
            pretrained_dict = self.state_dict()

        for k, v in self.named_parameters():
            if 'hm' in k:
                if 'bias' in k:
                    pretrained_dict[k] = torch.ones_like(v) * -math.log(
                        (1 - 0.01) / 0.01)
            print(k, v.requires_grad)

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        input('grad')
        for k, v in self.named_parameters():
            print(k, v.shape)
        input('model_parameters:')
        for k, v in pretrained_dict.items():
            print(k, v.shape)
        input('pretrained_parameters')


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    print(kwargs)
    input('s')
    if pretrained:
        model.init_weights(pretrained='/home/mayx/project/CenterNet/exp/cspdet/wider_resnet18_csp_multi/model_last.pth')
    print(model)
    input('s')
    return model


def resnet18(pretrained=True, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18',
                   BasicBlock, [2, 2, 2, 2],
                   pretrained,
                   progress,
                   replace_stride_with_dilation=[False, False, True],
                   **kwargs)


def resnet50(pretrained=True, progress=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50',
                   Bottleneck, [3, 4, 6, 3],
                   pretrained,
                   progress,
                   replace_stride_with_dilation=[False, False, True],
                   **kwargs)


def test():
    import torch
    net = resnet18(conv4=True,conv4_conv2=True,change_s1=True,all_gn=True)
    net=net.cuda()
    for k, v in net.named_parameters():
        print(k)    
    print(net)
    a = torch.ones(5, 3, 704, 704).cuda()
    b = net(a)
    c=b['hm'].sum()+b['hm_small'].sum()
    c.backward()
    for name, output in b.items():
        print(name, output.shape)
    input('s')


if __name__ == "__main__":
    test()
