import torch
import torchvision.models as models
from torch import nn, Tensor
from torch.nn.utils.weight_norm import WeightNorm


class BackboneClassifier(nn.Module):
    def __init__(self, n_classes, backbone, freeze_backbone):
        super(BackboneClassifier, self).__init__()
        self.backbone = backbone
        self.backbone.train()

        self.frozen_backbone = freeze_backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.n_features = self.backbone.fc.in_features
        self.n_classes = n_classes
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.n_features, self.n_classes),
        )

    def freeze_backbone(self):
        if not self.frozen_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        self.frozen_backbone = True

    def unfreeze_backbone(self):
        if self.frozen_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = True
        self.frozen_backbone = False

    def forward(self, x) -> Tensor:
        x = self.backbone(x)
        return x


class ResNet18Classifier(BackboneClassifier):
    def __init__(self, n_classes, freeze_backbone=False, pretrained=True):
        super(ResNet18Classifier, self).__init__(n_classes, models.resnet18(pretrained=pretrained),
                                                 freeze_backbone=freeze_backbone)


class ResNet34Classifier(BackboneClassifier):
    def __init__(self, n_classes, freeze_backbone=False, pretrained=True):
        super(ResNet34Classifier, self).__init__(n_classes, models.resnet34(pretrained=pretrained),
                                                 freeze_backbone=freeze_backbone)


class GoogLeNetClassifier(BackboneClassifier):
    def __init__(self, n_classes, freeze_backbone=False, pretrained=True):
        super(GoogLeNetClassifier, self).__init__(n_classes, models.googlenet(pretrained=pretrained),
                                                  freeze_backbone=freeze_backbone)


# class CosineClassifier(nn.Module):
#
#     def __init__(self, input_features, output_features, eps=10 ** -8):
#         super(CosineClassifier, self).__init__()
#         self.input_features = input_features
#         self.output_features = output_features
#         self.eps = eps
#
#         self.weights = nn.Parameter(torch.rand(size=(output_features, input_features)))
#         self.coefficients = nn.Parameter(torch.rand(size=(output_features,)))
#         self.scale = 2
#
#     def forward(self, x: Tensor):
#         x_len = x.pow(2).sum().rsqrt()
#         w_len = self.weights.pow(2).sum(keepdim=True, dim=1).rsqrt().squeeze()
#
#         return x.mm(self.weights.t()) * w_len * x_len * self.coefficients * self.scale
#
#     def extra_repr(self):
#         return 'in_features={}, out_features={}'.format(
#             self.input_features, self.output_features
#         )


class CosineClassifier(nn.Module):
    def __init__(self, input_features, output_features):
        super(CosineClassifier, self).__init__()
        self.L = nn.Linear(input_features, output_features, bias=False)
        self.class_wise_learnable_norm = True
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0)

        self.scale_factor = 2

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(
            x_normalized)
        scores = self.scale_factor * cos_dist

        return scores


class BackboneCosineClassifier(nn.Module):
    def __init__(self, n_classes, backbone, freeze_backbone):
        super(BackboneCosineClassifier, self).__init__()
        self.backbone = backbone
        self.backbone.train()

        self.frozen_backbone = freeze_backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.n_features = self.backbone.fc.in_features
        self.n_classes = n_classes
        self.backbone.fc = nn.Sequential(
            CosineClassifier(input_features=self.n_features, output_features=self.n_classes),
        )

    def freeze_backbone(self):
        if not self.frozen_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        self.frozen_backbone = True

    def unfreeze_backbone(self):
        if self.frozen_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = True
        self.frozen_backbone = False

    def forward(self, x) -> Tensor:
        x = self.backbone(x)
        return x


class ResNet18CosineClassifier(BackboneCosineClassifier):
    def __init__(self, n_classes, freeze_backbone=False, pretrained=True):
        super(ResNet18CosineClassifier, self).__init__(n_classes, models.resnet18(pretrained=pretrained),
                                                       freeze_backbone=freeze_backbone)

# clf = CosineClassifier(input_features=3, output_features=2)
# print(clf.weights, clf.coefficients)
# print(clf.weights[0], clf.weights[1])
# inp = torch.Tensor([list(clf.weights[0]), list(clf.weights[1])])
# print(clf(inp))
