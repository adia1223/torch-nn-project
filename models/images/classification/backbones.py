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

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)
        out = self.maxpool(out)
        return out


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
