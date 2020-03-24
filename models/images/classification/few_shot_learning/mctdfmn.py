import time
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from data import LABELED_DATASETS, LabeledSubdataset
from models.images.classification.backbones import NoPoolingBackbone
from models.images.classification.few_shot_learning import evaluate_solution, accuracy, FSLEpisodeSampler, \
    FEATURE_EXTRACTORS
from utils import pretty_time, remove_dim
from visualization.plots import PlotterWindow


class ScaleModule(nn.Module):
    def __init__(self, in_features, map_size):
        super(ScaleModule, self).__init__()
        self.in_features = in_features
        self.conv = nn.Conv2d(in_channels=self.in_features, out_channels=1, kernel_size=3)
        self.bn = nn.BatchNorm2d(1, eps=2e-5)
        self.relu = nn.ReLU()

        self.fc = nn.Linear((map_size - 2) ** 2, 1)
        self.sp = nn.Softplus()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.sp(x)

        return x


def lr_schedule(iter: int):
    if iter >= 35000:
        return 0.0012
    elif iter >= 25000:
        return 0.006
    else:
        return 0.1


class MCTDFMN(nn.Module):
    def __init__(self, backbone: NoPoolingBackbone, train_transduction_steps=1, test_transduction_steps=10, lmb=0.2):
        super(MCTDFMN, self).__init__()
        self.n_classes = None
        self.support_set_size = None
        self.support_set_features = None
        self.class_prototypes = None
        self.query_set_features = None
        self.query_set_size = None

        self.feature_extractor = backbone
        self.featmap_size = backbone.output_featmap_size()
        self.scale_module = ScaleModule(backbone.output_features(), self.featmap_size)
        self.feature_extractor.fc = nn.Sequential()

        self.train_ts = train_transduction_steps
        self.test_ts = test_transduction_steps

        self.loss_fn = nn.CrossEntropyLoss()
        self.lmb = lmb

    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(batch)
        return x

    def build_prototypes(self, support_set: torch.Tensor, query_set: torch.Tensor):
        its = self.train_ts if self.training else self.test_ts
        self.class_prototypes = torch.mean(support_set, dim=1)
        for i in range(its):
            self.class_prototypes = self.update_prototypes(support_set, query_set)

    def distance(self, a: torch.Tensor, b: torch.Tensor):
        a_scale = self.scale_module(a)
        b_scale = self.scale_module(b)

        a = a.reshape(a.size(0), -1)
        b = b.reshape(b.size(0), -1)

        a = F.normalize(a, dim=1)
        b = F.normalize(b, dim=1)
        a = torch.div(a, a_scale)
        b = torch.div(b, b_scale)
        return (a - b).pow(2).sum(dim=1)

    def l2_distance(self, a: torch.Tensor, b: torch.Tensor):
        return (a - b).pow(2).sum(dim=3)

    def get_proba(self, query_set: torch.Tensor):
        return F.softmax(self.get_distances(query_set), dim=1)

    def get_distances(self, query_set: torch.Tensor):
        query_set_expanded = query_set.repeat_interleave(self.n_classes, dim=0)
        prototypes_expanded = self.class_prototypes.repeat(self.query_set_size, 1, 1, 1)
        distances = self.distance(query_set_expanded, prototypes_expanded)
        distances = torch.stack(distances.split(self.n_classes))
        return -distances

    def get_l2_distances(self, query_set: torch.Tensor, prototypes: torch.Tensor):
        query_set_expanded = query_set.repeat_interleave(self.n_classes, dim=0)
        prototypes_expanded = prototypes.repeat(self.query_set_size, 1, 1, 1)
        distances = self.l2_distance(query_set_expanded, prototypes_expanded)
        distances = torch.stack(distances.split(self.n_classes))
        return -distances

    def update_prototypes(self, support_set: torch.Tensor, query_set: torch.Tensor):
        classes_denom = torch.tensor([self.support_set_size] * self.n_classes, device=query_set.device,
                                     dtype=torch.float)
        new_proto = torch.sum(support_set, dim=1)
        probas = self.get_proba(self.query_set_features)
        for cur_class in range(self.n_classes):
            class_probas = probas[:, cur_class].squeeze()
            classes_denom[cur_class] += class_probas.sum()
            new_proto[cur_class] += torch.mul(query_set, class_probas.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand_as(
                query_set)).sum(0)
        new_proto = torch.div(new_proto, classes_denom.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand_as(new_proto))
        return new_proto

    def forward(self, support_set: torch.Tensor, query_set: torch.Tensor) -> torch.Tensor:
        self.n_classes = support_set.size(0)
        self.support_set_size = support_set.size(1)
        self.query_set_size = query_set.size(0)

        self.support_set_features = self.extract_features(remove_dim(support_set, 1))

        self.support_set_features = self.support_set_features.view(
            *([self.n_classes, self.support_set_size] + list(self.support_set_features.shape)[1:]))

        self.query_set_features = self.extract_features(query_set)

        self.build_prototypes(self.support_set_features, self.query_set_features)

        return self.get_distances(self.query_set_features)

    def forward_with_loss(self, support_set: torch.Tensor, query_set: torch.Tensor,
                          labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self(support_set, query_set)
        loss_i = self.loss_fn(output, labels) * self.lmb
        query_set_features_t = self.query_set_features.permute(0, 2, 3, 1)
        prototypes_t = self.class_prototypes.permute(0, 2, 3, 1)
        pixel_wise_distances_t = self.get_l2_distances(query_set_features_t, prototypes_t)
        pixel_wise_distances = pixel_wise_distances_t.permute(0, 2, 3, 1)
        # print(self.query_set_size * (self.featmap_size ** 2), self.n_classes)
        labels_expanded = labels.repeat_interleave(repeats=self.featmap_size ** 2)
        pixel_wise_losses = F.cross_entropy(
            pixel_wise_distances.reshape(self.query_set_size * (self.featmap_size ** 2), self.n_classes),
            labels_expanded)
        # print(pixel_wise_losses)
        loss_d = pixel_wise_losses / self.query_set_size
        res_loss = loss_i + loss_d

        return output, res_loss


def train_mctdfmn(base_subdataset: LabeledSubdataset, val_subdataset: LabeledSubdataset, n_shot: int, n_way: int,
                  n_iterations: int, batch_size: int, eval_period: int,
                  train_n_way=15,
                  backbone_name='resnet12-np', lr=0.01,
                  device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), session_info=None):
    if session_info is None:
        session_info = dict()
    backbone = FEATURE_EXTRACTORS[backbone_name]()
    model = MCTDFMN(backbone=backbone).to(device)

    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, nesterov=True, weight_decay=0.0005, momentum=0.9)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)

    base_sampler = FSLEpisodeSampler(subdataset=base_subdataset, n_way=train_n_way, n_shot=n_shot,
                                     batch_size=batch_size)
    val_sampler = FSLEpisodeSampler(subdataset=val_subdataset, n_way=n_way, n_shot=n_shot, batch_size=batch_size)

    loss_plotter = PlotterWindow(interval=1000)
    accuracy_plotter = PlotterWindow(interval=1000)

    loss_plotter.new_line('Loss')
    accuracy_plotter.new_line('Train Accuracy')
    accuracy_plotter.new_line('Validation Accuracy')

    best_accuracy = 0
    best_iteration = -1

    print("Training started for parameters:")
    print(session_info)
    print()

    start_time = time.time()

    for iteration in range(n_iterations):
        model.train()

        support_set, batch = base_sampler.sample()
        query_set, query_labels = batch
        query_set = query_set.to(device)
        query_labels = query_labels.to(device)

        optimizer.zero_grad()
        output, loss = model.forward_with_loss(support_set, query_set, query_labels)
        # output = model.forward(support_set, query_set)
        # loss = loss_fn(output, query_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        labels_pred = output.argmax(dim=1)
        labels = query_labels
        cur_accuracy = accuracy(labels=labels, labels_pred=labels_pred)

        loss_plotter.add_point('Loss', iteration, loss.item())
        accuracy_plotter.add_point('Train Accuracy', iteration, cur_accuracy)

        if iteration % eval_period == 0 or iteration == n_iterations - 1:
            val_start_time = time.time()

            val_accuracy = evaluate_solution(model, val_sampler)
            accuracy_plotter.add_point('Validation Accuracy', iteration, val_accuracy)
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_iteration = iteration
                print("Best evaluation result yet!")

            cur_time = time.time()

            val_time = cur_time - val_start_time
            time_used = cur_time - start_time
            time_per_iteration = time_used / (iteration + 1)

            print()
            print("[%d/%d] = %.2f%%\t\tLoss: %.4f" % (
                iteration + 1, n_iterations, (iteration + 1) / n_iterations * 100, loss.item()))
            print("Current validation time: %s" % pretty_time(val_time))

            print('Average iteration time: %s\tEstimated execution time: %s' % (
                pretty_time(time_per_iteration),
                pretty_time(time_per_iteration * (n_iterations - iteration - 1)),
            ))
            print()

    cur_time = time.time()
    training_time = cur_time - start_time
    print("Training finished. Total execution time: %s" % pretty_time(training_time))
    print("Best accuracy is: %.3f" % best_accuracy)
    print("Best iteration is: [%d/%d]" % (best_iteration + 1, n_iterations))
    print()

    session_info['accuracy'] = best_accuracy
    session_info['best_iteration'] = best_iteration
    session_info['execution_time'] = training_time


if __name__ == '__main__':
    print("Preparations for training...")
    dataset = LABELED_DATASETS['gtsrb'](augment_prob=0.5)
    base_subdataset, val_subdataset = dataset.subdataset.extract_classes(20)
    base_subdataset.set_test(False)
    val_subdataset.set_test(True)
    train_mctdfmn(base_subdataset, val_subdataset, 5, 5, 40000, 16, 1000)
