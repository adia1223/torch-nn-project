import torchvision
from torch import nn


class Resnet18NoPooling(nn.Module):
    def __init__(self, pretrained=False):
        super(Resnet18NoPooling, self).__init__()
        self.backbone = torchvision.models.resnet18(pretrained=pretrained)
        # self.backbone.fc = nn.Sequential()
        # self.backbone.avgpool = nn.Sequential()

        self.fc = nn.Linear(in_features=512, out_features=1)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        return x
