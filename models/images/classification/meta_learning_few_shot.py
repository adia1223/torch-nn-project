import random

import torch
from torch import nn
from torch.nn.utils.weight_norm import WeightNorm
from torch.optim.optimizer import Optimizer

from torch_utils import combine_dimensions


class FewShotLearningTask(nn.Module):
    def __init__(self):
        super(FewShotLearningTask, self).__init__()
        self.loss = None

    def process_support_set(self, support_set):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_loss(self, x, y, support_set=None):
        if support_set is not None:
            self.process_support_set(support_set)
        y_pred = self(x)
        return self.loss(y_pred, y), y_pred

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def train_step(self, x: torch.Tensor, y: torch.Tensor, support_set, optimizer: Optimizer):
        optimizer.zero_grad()
        loss, y_pred = self.compute_loss(x, y, support_set)
        loss.backward()
        optimizer.step()
        return loss, y_pred

    def test_step(self, x: torch.Tensor, y: torch.Tensor, support_set):
        loss, y_pred = self.compute_loss(x, y, support_set)

        return loss, y_pred


class BackBoneBasedModel:
    def __init__(self, backbone):
        self.backbone = backbone
        self.backbone_features = backbone.fc.in_features
        self.backbone.train()
        self.frozen_backbone = False

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


class SupportSetMeanFeaturesModel(FewShotLearningTask, BackBoneBasedModel):
    def __init__(self, extractor):
        FewShotLearningTask.__init__(self)
        BackBoneBasedModel.__init__(self, extractor)

        self.backbone.fc = nn.Sequential()

        self.n_classes = None
        self.support_set_size = None
        self.classes_features = None

    def process_support_set(self, support_set):
        self.n_classes = len(support_set)

        images = []

        class_size = []

        for label in range(self.n_classes):
            class_support_set = support_set[label]
            class_size.append(len(class_support_set))
            for image in class_support_set:
                images.append(image)

        batch = torch.stack(images, dim=0)
        features = self.backbone(batch)
        # print(features.shape)

        self.classes_features = torch.stack(torch.split(features, class_size, dim=0))
        # print(self.classes_features.shape)
        self.classes_features = torch.mean(self.classes_features, dim=1)
        # self.classes_features = torch.div(self.classes_features,
        #                                   torch.tensor(
        #                                       class_size,
        #                                       device=self.classes_features.device
        #                                   ).float().view(-1, 1).repeat_interleave(repeats=self.classes_features.size(1),
        #                                                                           dim=1)
        #                                   )

        # print(self.classes_features.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ProtoNetClassifier(SupportSetMeanFeaturesModel):
    def __init__(self, backbone, *args, **kwargs):
        super(ProtoNetClassifier, self).__init__(backbone)

        self.softmax = nn.Softmax()

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        batch_size = x.size()[0]
        x = x.view(batch_size, 1, -1).repeat_interleave(repeats=self.n_classes, dim=1)

        dist = torch.sum((self.classes_features.unsqueeze(0).repeat_interleave(repeats=batch_size, dim=0) - x).pow(2),
                         dim=2)

        return dist * -1

    def predict(self, x):
        y_pred = self(x)
        result = self.softmax(y_pred)

        return result


class RelationConvBlock(nn.Module):
    def __init__(self, indim, outdim, padding=0):
        super(RelationConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
        self.BN = nn.BatchNorm2d(outdim, momentum=1, affine=True)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

        self.parametrized_layers = [self.C, self.BN, self.relu, self.pool]

        # for layer in self.parametrized_layers:
        #     backbone.init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out


class RelationModule(nn.Module):

    def __init__(self, n_features, n_hidden=32):
        super(RelationModule, self).__init__()
        self.n_features = n_features

        self.conv1 = RelationConvBlock(n_features * 2, n_features, padding=1)
        self.conv2 = RelationConvBlock(n_features, n_features, padding=1)

        self.flattened_features = n_features

        self.fc1 = nn.Sequential(
            nn.Linear(self.flattened_features, n_hidden),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x: torch.Tensor):
        # print(x.size())
        x = self.conv1(x)
        # print(x.size())
        x = self.conv2(x)
        # print(x.size())

        x = x.view(x.size(0), -1)
        # print(x.size())

        x = self.fc1(x)
        x = self.fc2(x)
        return x


class RelationNetClassifier(SupportSetMeanFeaturesModel):
    def __init__(self, backbone, *args, **kwargs):
        super(RelationNetClassifier, self).__init__(backbone)

        self.relation_module = RelationModule(self.backbone_features)
        self.softmax = nn.Softmax()

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        # print(x.shape, self.classes_features.shape)
        batch_size = x.size()[0]
        classes_cnt = self.classes_features.size()[0]
        x = x.unsqueeze(1).repeat_interleave(repeats=self.n_classes, dim=1)

        support_set = self.classes_features.unsqueeze(0).repeat_interleave(repeats=batch_size, dim=0)
        # print(x.shape, support_set.shape)
        # print(torch.cat((x, support_set), dim=2).shape)
        # concatenated = torch.cat((x, support_set), dim=2).view(batch_size * classes_cnt, -1)

        concatenated = combine_dimensions(torch.cat((x, support_set), dim=2), 0, 2)

        relations = self.relation_module(concatenated)
        return relations.view(batch_size, classes_cnt)

    def predict(self, x):
        y_pred = self(x)
        result = self.softmax(y_pred)

        return result


class BaselineClassifier(FewShotLearningTask, BackBoneBasedModel):
    def __init__(self, backbone):
        FewShotLearningTask.__init__(self)
        BackBoneBasedModel.__init__(self, backbone)

        self.n_classes = None
        self.softmax = nn.Softmax()

        self.loss = nn.CrossEntropyLoss()

    def process_support_set(self, support_set, n_iterations=100, batch_size=4):
        self.n_classes = len(support_set)
        # print(self)
        self.backbone.fc = self.classifier(self.backbone_features, self.n_classes).to(support_set[0][0].device)
        # print(self)
        images = []
        labels = []

        for label in range(self.n_classes):
            class_support_set = support_set[label]
            for image in class_support_set:
                images.append(image)
            labels += [label] * len(class_support_set)

        images = torch.stack(images, dim=0)

        device = images.device

        labels = torch.tensor(labels, device=device)

        support_set_size = labels.size()[0]
        batch_size = min(batch_size, support_set_size)

        optimizer = torch.optim.SGD(self.backbone.fc.parameters(), lr=0.01, momentum=0.9, dampening=0.9,
                                    weight_decay=0.001)

        for iteration in range(n_iterations):
            indices = torch.tensor(random.sample(list(range(support_set_size)), batch_size), device=device)
            x = images.index_select(0, indices)
            y = labels.index_select(0, indices)

            optimizer.zero_grad()
            y_pred = self(x)
            loss = self.loss(y_pred, y)
            # print(self)
            # print(loss)
            loss.backward()
            optimizer.step()

    def classifier(self, in_features, out_features) -> nn.Module:
        return nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        y_pred = self(x)
        result = self.softmax(y_pred)

        return result


class CosineClassifier(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineClassifier, self).__init__()
        self.L = nn.Linear(in_features, out_features, bias=False)
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


class BaselinePlusPlusClassifier(BaselineClassifier):
    def classifier(self, in_features, out_features) -> nn.Module:
        return CosineClassifier(in_features=in_features, out_features=out_features)


MODELS = {
    'Baseline': BaselineClassifier,
    'Baseline++': BaselinePlusPlusClassifier,
    'ProtoNet': ProtoNetClassifier,
    'RelationNet': RelationNetClassifier,
}
