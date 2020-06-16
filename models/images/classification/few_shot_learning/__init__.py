import random
import time

import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data.dataset import Dataset
from torchvision import models

from data import LabeledSubdataset
from models.images.classification.backbones import ResNet18NoFlattening, ResNet12NoFlattening, \
    ResNet12NoFlatteningOriginal, \
    ConvNet256Original, ConvNet64Original, ConvNet64PoolingOriginal
from utils import remove_dim, pretty_time


class FewShotLearningSolution(nn.Module):
    def __init__(self):
        super(FewShotLearningSolution, self).__init__()

    def forward(self, support_set: torch.Tensor, query_set: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ProtoNetBasedFSLSolution(FewShotLearningSolution):
    def __init__(self):
        super(ProtoNetBasedFSLSolution, self).__init__()

        self.n_classes = None
        self.support_set_size = None
        self.support_set_features = None
        self.class_prototypes = None
        self.query_set_features = None
        self.query_set_size = None

    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_prototypes(self, support_set: torch.Tensor, query_set: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, support_set: torch.Tensor, query_set: torch.Tensor) -> torch.Tensor:
        self.n_classes = support_set.size(0)
        self.support_set_size = support_set.size(1)
        self.query_set_size = query_set.size(0)

        self.support_set_features = self.extract_features(remove_dim(support_set, 1)).view(self.n_classes,
                                                                                           self.support_set_size, -1)

        self.query_set_features = self.extract_features(query_set)

        self.class_prototypes = self.get_prototypes(self.support_set_features, self.query_set_features)

        query_set_features_prepared = self.query_set_features.unsqueeze(1).repeat_interleave(repeats=self.n_classes,
                                                                                             dim=1)

        distance = torch.sum((self.class_prototypes.unsqueeze(0).repeat_interleave(repeats=self.query_set_size,
                                                                                   dim=0) -
                              query_set_features_prepared).pow(2), dim=2)

        return -distance


def accuracy(labels, labels_pred):
    return accuracy_score(labels_pred.cpu(), labels.cpu())


def adam(model: nn.Module, lr=0.001):
    return torch.optim.Adam(model.parameters(), lr=lr)


def sgd(model: nn.Module, lr=0.001):
    return torch.optim.SGD(model.parameters(), lr=lr)


OPTIMIZERS = {
    'adam': adam,
    'sgd': sgd,
}

FEATURE_EXTRACTORS = {
    'resnet18': lambda: models.resnet18(pretrained=False),
    'googlenet': lambda: models.googlenet(pretrained=False),
    'resnet18-np': lambda: ResNet18NoFlattening(pretrained=False),
    'resnet12-np': lambda: ResNet12NoFlattening(),
    'resnet12-np-o': lambda: ResNet12NoFlatteningOriginal(),
    'conv256-np-o': lambda: ConvNet256Original(),
    'conv64-np-o': lambda: ConvNet64Original(),
    'conv64-p-o': lambda: ConvNet64PoolingOriginal(),
}


class FSLEpisodeSampler(Dataset):
    def __init__(self, subdataset: LabeledSubdataset, n_way: int, n_shot: int, batch_size: int, balanced: bool,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.subdataset = subdataset
        self.n_way = n_way
        self.n_shot = n_shot
        self.batch_size = batch_size
        self.device = device
        self.balanced = balanced

    def sample(self):
        cur_subdataset, _ = self.subdataset.extract_classes(self.n_way)
        support_subdataset, query_subdataset = cur_subdataset.extract_balanced(self.n_shot)
        classes_mapping = {}

        support_set_labels = support_subdataset.labels()

        support_set = [[] for i in range(len(support_set_labels))]

        h = 0
        for label in support_set_labels:
            classes_mapping[label] = h
            h += 1

        for i in range(len(support_subdataset)):
            item, label, _ = support_subdataset[i]
            support_set[classes_mapping[label]].append(item.to(self.device))
        if not self.balanced:
            batch = query_subdataset.random_batch(self.batch_size)
        else:
            batch = query_subdataset.balanced_batch(self.batch_size)

        for i in range(len(batch[1])):
            batch[1][i] = classes_mapping[batch[1][i].item()]

        for i in range(len(support_set)):
            support_set[i] = torch.stack(support_set[i])
        support_set = torch.stack(support_set)

        return support_set, batch


class FSLEpisodeSamplerGlobalLabels(FSLEpisodeSampler):
    def sample(self):
        cur_subdataset, _ = self.subdataset.extract_classes(self.n_way)
        support_subdataset, query_dataset = cur_subdataset.extract_balanced(self.n_shot)
        classes_mapping = {}

        support_set_labels = support_subdataset.labels()

        support_set = [[] for i in range(len(support_set_labels))]

        h = 0
        for label in support_set_labels:
            classes_mapping[label] = h
            h += 1

        for i in range(len(support_subdataset)):
            item, label, _ = support_subdataset[i]
            support_set[classes_mapping[label]].append(item.to(self.device))
        if not self.balanced:
            batch = list(query_dataset.random_batch(self.batch_size))
        else:
            batch = list(query_dataset.balanced_batch(self.batch_size))
        # batch = list(cur_subdataset.random_batch(self.batch_size))
        # batch.append([-1] * len(batch[1]))
        for i in range(len(batch[1])):
            # batch[2][i] = batch[1][i].item()
            batch[1][i] = classes_mapping[batch[1][i].item()]
        # batch[2] = torch.tensor(batch[2])

        for i in range(len(support_set)):
            support_set[i] = torch.stack(support_set[i])
        support_set = torch.stack(support_set)

        return support_set, batch, classes_mapping


class TripletBatchSampler:
    def __init__(self, subdataset: LabeledSubdataset, batch_size: int,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.subdataset = subdataset
        self.batch_size = batch_size
        self.device = device

    def sample(self):
        anchor = []
        positive = []
        negative = []
        for i in range(self.batch_size):
            selected_class, other_classes = self.subdataset.extract_classes(1)
            anchor_positive, _ = selected_class.random_batch(2)
            anchor_positive = list(anchor_positive)
            random.shuffle(anchor_positive)
            anchor.append(anchor_positive[0].to(self.device))
            positive.append(anchor_positive[1].to(self.device))

            negative_, _ = other_classes.random_batch(1)
            negative.append(negative_[0].to(self.device))

        return torch.stack(anchor), torch.stack(positive), torch.stack(negative)


def evaluate_solution_episodes(model: FewShotLearningSolution, validation_sampler: FSLEpisodeSampler,
                               n_iterations=600, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    val_start_time = time.time()

    print("Evaluation started...")
    model = model.to(device)
    model.eval()
    res_accuracy = 0
    with torch.no_grad():
        for i in range(n_iterations):
            support_set, batch = validation_sampler.sample()
            query_set, query_labels = batch
            query_set = query_set.to(device)
            query_labels = query_labels.to(device)

            output = model.forward(support_set, query_set)
            labels_pred = output.argmax(dim=1)
            labels = query_labels
            cur_accuracy = accuracy(labels=labels, labels_pred=labels_pred)
            res_accuracy += cur_accuracy
    res_accuracy /= n_iterations
    cur_time = time.time()
    val_time = cur_time - val_start_time
    print("Evaluation completed: accuracy = %.3f" % res_accuracy)
    print("Evaluation time: %s" % pretty_time(val_time))

    return res_accuracy
