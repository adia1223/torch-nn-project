import copy
import os
import random
import time
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from data import LABELED_DATASETS, LabeledSubdataset
from models.images.classification.backbones import NoFlatteningBackbone
from models.images.classification.few_shot_learning import evaluate_solution_episodes, accuracy, FSLEpisodeSampler, \
    FEATURE_EXTRACTORS, FSLEpisodeSamplerGlobalLabels
from sessions import Session
from utils import pretty_time, remove_dim, inverse_mapping
from visualization.plots import PlotterWindow

MAX_BATCH_SIZE = 500

EPOCHS_MULTIPLIER = 1


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
    if iter >= 30000 * EPOCHS_MULTIPLIER:
        return 0.0012
    elif iter >= 20000 * EPOCHS_MULTIPLIER:
        return 0.006
    else:
        return 0.1


class MCTDFMN(nn.Module):
    def __init__(self, train_classes: int, backbone: NoFlatteningBackbone, train_transduction_steps=1,
                 test_transduction_steps=10, lmb=0.2, all_global_prototypes=True,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super(MCTDFMN, self).__init__()
        self.n_classes = None
        self.support_set_size = None
        self.support_set_features = None
        self.class_prototypes = None
        self.query_set_features = None
        self.query_set_size = None
        self.device = device

        self.train_classes = train_classes
        self.all_global_prototypes = all_global_prototypes

        self.feature_extractor = backbone
        self.featmap_size = backbone.output_featmap_size()
        self.featmap_size2 = self.featmap_size ** 2
        self.scale_module = ScaleModule(backbone.output_features(), self.featmap_size)
        self.global_proto = nn.Linear(in_features=backbone.output_features(), out_features=train_classes)

        nn.init.xavier_uniform_(self.global_proto.weight)

        self.train_ts = train_transduction_steps
        self.test_ts = test_transduction_steps

        self.loss_fn = nn.CrossEntropyLoss()
        self.lmb = lmb

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

    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        # print(batch.size())
        minibatches = batch.split(split_size=MAX_BATCH_SIZE)
        # print(minibatches)
        xs = []
        for minibatch in minibatches:
            xs.append(self.feature_extractor(minibatch))
            # print(len(xs))
            # print(xs[-1].size())
        x = torch.cat(xs)
        # print(x.size())
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
        return (a - b).pow(2).sum(dim=1)

    def get_proba(self, query_set: torch.Tensor):
        return F.softmax(self.get_distances(query_set), dim=1)

    def get_distances(self, query_set: torch.Tensor):
        query_set_expanded = query_set.repeat_interleave(self.n_classes, dim=0)
        prototypes_expanded = self.class_prototypes.repeat(self.query_set_size, 1, 1, 1)
        distances = self.distance(query_set_expanded, prototypes_expanded)
        distances = torch.stack(distances.split(self.n_classes))
        return -distances

    def get_l2_distances(self, query_set: torch.Tensor, prototypes: torch.Tensor):
        cur_n_classes = prototypes.size(0)
        cur_query_set_size = query_set.size(0)

        query_set_expanded = query_set.repeat_interleave(cur_n_classes, dim=0)
        prototypes_expanded = prototypes.repeat(cur_query_set_size, 1)
        distances = self.l2_distance(query_set_expanded, prototypes_expanded)
        distances = torch.stack(distances.split(cur_n_classes))
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
                          labels: torch.Tensor, global_classes_mapping: dict) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        output = self(support_set, query_set)
        loss_i = self.loss_fn(output, labels)

        cur_labels = labels.clone().repeat_interleave(self.featmap_size2, dim=0)
        cur_global_prototypes = self.global_proto.weight
        inv_mapping = inverse_mapping(global_classes_mapping)
        if self.all_global_prototypes:
            for i in range(cur_labels.size(0)):
                cur_labels[i] = inv_mapping[cur_labels[i].item()]
        else:
            indices = []
            for i in range(support_set.size(0)):
                indices.append(inv_mapping[i])
            indices = torch.tensor(indices, device=self.device)
            cur_global_prototypes = torch.index_select(cur_global_prototypes, 0, indices)
        # print(cur_labels.size())

        expanded_global_prototypes = cur_global_prototypes
        # expanded_query_set = torch.reshape(self.query_set_features, (self.query_set_features.size(0), -1))
        expanded_query_set = self.query_set_features.permute(0, 2, 3, 1).reshape((-1, self.query_set_features.size(1)))
        # print(expanded_query_set.shape)
        # print(expanded_global_prototypes.shape)
        d_distances = self.get_l2_distances(expanded_query_set, expanded_global_prototypes)
        # print(d_distances.shape)
        loss_d = self.loss_fn(d_distances, cur_labels)  # * self.featmap_size2
        # print(loss_d.item(), loss_i.item())

        res_loss = (0.2 * loss_i) + loss_d

        return output, res_loss, loss_i, loss_d


def train_mctdfmn(base_subdataset: LabeledSubdataset, val_subdataset: LabeledSubdataset, n_shot: int, n_way: int,
                  n_iterations: int, batch_size: int, eval_period: int,
                  val_batch_size: int,
                  dataset_classes: int,
                  image_size: int,
                  balanced_batches: bool,
                  train_n_way=15,
                  backbone_name='resnet12-np', lr=0.1,
                  train_ts_steps=1,
                  test_ts_steps=10,
                  all_global_prototypes=True,
                  device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), **kwargs):
    session_info = {
        "task": "few-shot learning",
        "model": "MCT_DFMN",
        "feature_extractor": backbone_name,
        "n_iterations": n_iterations,
        "eval_period": eval_period,
        # "dataset": dataset_name,
        # "optimizer": optimizer_name,
        "batch_size": batch_size,
        "val_batch_size": val_batch_size,
        "n_shot": n_shot,
        "n_way": n_way,
        "train_n_way": train_n_way,
        "train_ts_steps": train_ts_steps,
        "test_ts_steps": test_ts_steps,
        "optimizer": 'sgd',
        "all_global_prototypes": all_global_prototypes,
        "image_size": image_size,
        "balanced_batches": balanced_batches,
    }

    session_info.update(kwargs)

    backbone = FEATURE_EXTRACTORS[backbone_name]()
    model = MCTDFMN(backbone=backbone, test_transduction_steps=test_ts_steps,
                    train_transduction_steps=train_ts_steps, train_classes=dataset_classes,
                    all_global_prototypes=all_global_prototypes).to(device)

    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, nesterov=True, weight_decay=0.0005, momentum=0.9)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)

    base_sampler = FSLEpisodeSamplerGlobalLabels(subdataset=base_subdataset, n_way=train_n_way, n_shot=n_shot,
                                                 batch_size=batch_size, balanced=balanced_batches)
    val_sampler = FSLEpisodeSampler(subdataset=val_subdataset, n_way=n_way, n_shot=n_shot, batch_size=val_batch_size,
                                    balanced=balanced_batches)

    loss_plotter = PlotterWindow(interval=1000)
    accuracy_plotter = PlotterWindow(interval=1000)

    loss_plotter.new_line('Loss')
    loss_plotter.new_line('Dense Loss')
    loss_plotter.new_line('Instance Loss')
    accuracy_plotter.new_line('Train Accuracy')
    accuracy_plotter.new_line('Validation Accuracy')

    losses = []
    losses_d = []
    losses_i = []
    acc_train = []
    acc_val = []
    val_iters = []

    best_model = copy.deepcopy(model)

    best_accuracy = 0
    best_iteration = -1

    print("Training started for parameters:")
    print(session_info)
    print()

    start_time = time.time()

    for iteration in range(n_iterations):
        model.train()

        support_set, batch, global_classes_mapping = base_sampler.sample()
        # print(support_set.size())
        query_set, query_labels = batch
        # print(query_set.size())
        # print(global_classes_mapping)
        query_set = query_set.to(device)
        query_labels = query_labels.to(device)

        optimizer.zero_grad()
        output, loss, loss_i, loss_d = model.forward_with_loss(support_set, query_set, query_labels,
                                                               global_classes_mapping)
        # output = model.forward(support_set, query_set)
        # loss = loss_fn(output, query_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        labels_pred = output.argmax(dim=1)
        labels = query_labels
        cur_accuracy = accuracy(labels=labels, labels_pred=labels_pred)

        loss_plotter.add_point('Loss', iteration, loss.item())
        loss_plotter.add_point('Dense Loss', iteration, loss_d.item())
        loss_plotter.add_point('Instance Loss', iteration, loss_i.item())
        accuracy_plotter.add_point('Train Accuracy', iteration, cur_accuracy)

        losses.append(loss.item())
        losses_i.append(loss_i.item())
        losses_d.append(loss_d.item())
        acc_train.append(cur_accuracy)

        if iteration % eval_period == 0 or iteration == n_iterations - 1:
            val_start_time = time.time()

            val_accuracy = evaluate_solution_episodes(model, val_sampler)
            accuracy_plotter.add_point('Validation Accuracy', iteration, val_accuracy)

            acc_val.append(val_accuracy)
            val_iters.append(iteration + 1)

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_iteration = iteration
                best_model = copy.deepcopy(model)
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

    session = Session()
    session.build(name="FSL_MCTDFMN", comment=r"Few-Shot Learning solution based on https://arxiv.org/abs/2002.12017",
                  **session_info)
    # session.data.update(session_info)
    # save_record(name="Few-Shot Learning Training: MCT + DFMN", **session_info)

    torch.save(best_model, os.path.join(session.data['output_dir'], "trained_model_state_dict.tar"))
    iters = list(range(1, n_iterations + 1))

    plt.figure(figsize=(20, 20))
    plt.plot(iters, losses, label="Loss")
    plt.plot(iters, losses_d, label="Dense Loss")
    plt.plot(iters, losses_i, label="Instance Loss")
    plt.legend()
    plt.savefig(os.path.join(session.data['output_dir'], "loss_plot.png"))

    plt.figure(figsize=(20, 20))
    plt.plot(iters, acc_train, label="Train Accuracy")
    plt.plot(val_iters, acc_val, label="Test Accuracy")
    plt.legend()
    plt.savefig(os.path.join(session.data['output_dir'], "acc_plot.png"))

    session.save_info()


if __name__ == '__main__':
    torch.random.manual_seed(2002)
    random.seed(2002)

    DATASET_NAME = 'google-landmarks'
    BASE_CLASSES = 3000
    AUGMENT_PROB = 1.0
    ITERATIONS = 40000 * EPOCHS_MULTIPLIER
    N_WAY = 15
    EVAL_PERIOD = 1000
    RECORD = 730
    ALL_GLOBAL_PROTOTYPES = False
    IMAGE_SIZE = 84
    BACKBONE = 'conv64-np-o'
    BATCH_SIZE = 5 // EPOCHS_MULTIPLIER
    VAL_BATCH_SIZE = 5 // EPOCHS_MULTIPLIER
    BALANCED_BATCHES = True

    # N_SHOT = 5

    print("Preparations for training...")
    dataset = LABELED_DATASETS[DATASET_NAME](augment_prob=AUGMENT_PROB, image_size=IMAGE_SIZE)
    base_subdataset, val_subdataset = dataset.subdataset.extract_classes(BASE_CLASSES)
    base_subdataset.set_test(False)
    val_subdataset.set_test(True)

    for N_SHOT in (1,):
        train_mctdfmn(base_subdataset=base_subdataset, val_subdataset=val_subdataset, n_shot=N_SHOT, n_way=N_WAY,
                      n_iterations=ITERATIONS, batch_size=BATCH_SIZE,
                      eval_period=EVAL_PERIOD,
                      record=RECORD,
                      augment=AUGMENT_PROB,
                      dataset=DATASET_NAME,
                      base_classes=BASE_CLASSES,
                      dataset_classes=dataset.CLASSES,
                      all_global_prototypes=ALL_GLOBAL_PROTOTYPES,
                      image_size=IMAGE_SIZE,
                      backbone_name=BACKBONE,
                      balanced_batches=BALANCED_BATCHES,
                      val_batch_size=VAL_BATCH_SIZE,
                      train_ts_steps=0,
                      test_ts_steps=0)
