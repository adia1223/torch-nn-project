import torchvision
from torch import nn
from torchvision.models.resnet import BasicBlock


class NoPoolingBackbone(nn.Module):
    def __init__(self):
        super(NoPoolingBackbone, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def output_featmap_size(self):
        raise NotImplementedError

    def output_features(self):
        raise NotImplementedError


class ResNet18NoPooling(NoPoolingBackbone):

    def __init__(self, pretrained=False, drop_ratio=0.1):
        super(ResNet18NoPooling, self).__init__()
        self.backbone = torchvision.models.resnet18(pretrained=pretrained)
        self.dropout = nn.Dropout(drop_ratio)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='conv2d')
                try:
                    nn.init.constant_(m.bias, 0)
                except AttributeError as e:
                    pass

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.dropout(x)
        x = self.backbone.layer2(x)
        x = self.dropout(x)
        x = self.backbone.layer3(x)
        x = self.dropout(x)
        x = self.backbone.layer4(x)
        x = self.dropout(x)

        return x

    def output_featmap_size(self):
        return 3

    def output_features(self):
        return 512


# class BaseResNet18


class ResNet12NoPooling(NoPoolingBackbone):
    def __init__(self, drop_ratio=0.1):
        super(ResNet12NoPooling, self).__init__()
        self.backbone = torchvision.models.ResNet(BasicBlock, [1, 1, 1, 1])
        self.dropout = nn.Dropout(drop_ratio)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='conv2d')
                try:
                    nn.init.constant_(m.bias, 0)
                except AttributeError as e:
                    pass

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.dropout(x)
        x = self.backbone.layer2(x)
        x = self.dropout(x)
        x = self.backbone.layer3(x)
        x = self.dropout(x)
        x = self.backbone.layer4(x)
        x = self.dropout(x)
        # print(x.size())

        return x

    def output_featmap_size(self):
        return 3

    def output_features(self):
        return 512


ceil = True
inp = False


class ResNetBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(planes, eps=2e-5)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes, eps=2e-5)
        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(planes, eps=2e-5)

        self.convr = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, padding=1)
        self.bnr = nn.BatchNorm2d(planes, eps=2e-5)

        self.relu = nn.ReLU(inplace=inp)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil)

    def forward(self, x):
        identity = self.convr(x)
        identity = self.bnr(identity)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += identity
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class ResNet12NoPoolingOriginal(NoPoolingBackbone):
    def __init__(self, drop_ratio=0.1, with_drop=True):
        super(ResNet12NoPoolingOriginal, self).__init__()

        self.drop_layers = with_drop
        self.inplanes = 3
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_ratio, inplace=inp)
        self.layer1 = self._make_layer(ResNetBlock, 64)
        self.layer2 = self._make_layer(ResNetBlock, 128)
        self.layer3 = self._make_layer(ResNetBlock, 256)
        self.layer4 = self._make_layer(ResNetBlock, 512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='conv2d')
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes):
        layers = [block(self.inplanes, planes)]
        self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)

        x = self.layer2(x)
        x = self.dropout(x)

        x = self.layer3(x)
        x = self.dropout(x)

        x = self.layer4(x)
        x = self.dropout(x)

        # if self.drop_layers:
        #     return [x4, x3]
        # else:
        #     return [x4]

        # print(x.shape)

        return x

    def output_featmap_size(self):
        return 6

    def output_features(self):
        return 512


def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight)  # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    )


class ConvNet256Original(NoPoolingBackbone):

    def __init__(self, with_drop=False):
        super().__init__()

        self.drop_layer = with_drop

        self.hidden = 64
        self.layer1 = conv_block(3, self.hidden)
        self.layer2 = conv_block(self.hidden, int(1.5 * self.hidden))
        self.layer3 = conv_block(int(1.5 * self.hidden), 2 * self.hidden)
        self.layer4 = conv_block(2 * self.hidden, 4 * self.hidden)

        # self.weight = nn.Linear(4 * self.hidden, 64)
        # nn.init.xavier_uniform_(self.weight.weight)
        #
        # self.conv1_ls = nn.Conv2d(in_channels=4 * self.hidden, out_channels=1, kernel_size=3)
        # self.bn1_ls = nn.BatchNorm2d(1, eps=2e-5)
        # self.relu = nn.ReLU(inplace=True)
        # self.fc1_ls = nn.Linear(16, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.size())
        return x

    def output_featmap_size(self):
        return 6

    def output_features(self):
        return 256


class ConvNet64Original(NoPoolingBackbone):

    def __init__(self, with_drop=False, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()

        self.drop_layer = with_drop

        self.layer1 = conv_block(x_dim, hid_dim)
        self.layer2 = conv_block(hid_dim, hid_dim)
        self.layer3 = conv_block(hid_dim, z_dim)
        self.layer4 = conv_block(z_dim, z_dim)

        # # global weight
        # self.weight = nn.Linear(z_dim, 64)
        # nn.init.xavier_uniform_(self.weight.weight)
        #
        # # length scale parameters
        # self.conv1_ls = nn.Conv2d(in_channels=z_dim, out_channels=1, kernel_size=3)
        # self.bn1_ls = nn.BatchNorm2d(1, eps=2e-5)
        # self.relu = nn.ReLU(inplace=True)
        # self.fc1_ls = nn.Linear(16, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.size())
        return x

    def output_featmap_size(self):
        return 6

    def output_features(self):
        return 64

# if __name__ == '__main__':
#     orig = ResNet12NoPoolingOriginal()
#     my = ResNet12NoPooling()
#     print(orig)
#     print(my)
