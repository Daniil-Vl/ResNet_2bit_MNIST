import torch
import torch.nn.functional as F
from modules import ActFn, Conv2d, Linear
from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44']


BITWIDTH = 2


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)


class LambdaLayer(torch.nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        # self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                            padding=1, bias=False, bitwidth=BITWIDTH)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.alpha1 = torch.nn.Parameter(torch.tensor(10.))
        # self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1,
                            padding=1, bias=False, bitwidth=BITWIDTH)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.alpha2 = torch.nn.Parameter(torch.tensor(10.))
        self.ActFn = ActFn.apply
        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(lambda x: F.pad(
                x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))

    def forward(self, x):
        out = self.ActFn(self.bn1(self.conv1(x)), self.alpha1, BITWIDTH)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.ActFn(out, self.alpha2, BITWIDTH)
        return out


class ResNet(torch.nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = Conv2d(1, 16, kernel_size=3, stride=1,
                            padding=1, bias=False, bitwidth=BITWIDTH)

        self.bn1 = torch.nn.BatchNorm2d(16)
        self.alpha1 = torch.nn.Parameter(torch.tensor(10.))
        self.ActFn = ActFn.apply
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = Linear(64, num_classes, bitwidth=BITWIDTH)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.ActFn(self.bn1(self.conv1(x)), self.alpha1, BITWIDTH)
        # print("Forward in Resnet")
        # print(out.unique())
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(
        lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
