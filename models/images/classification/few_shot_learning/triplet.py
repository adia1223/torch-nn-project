import os
import random
import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn

from data import LABELED_DATASETS, LabeledSubdataset
from models.images.classification.backbones import NoFlatteningBackbone
from models.images.classification.few_shot_learning import evaluate_solution_episodes, FSLEpisodeSampler, \
    FEATURE_EXTRACTORS, OPTIMIZERS, TripletBatchSampler
from sessions import Session
from utils import pretty_time, remove_dim
from visualization.plots import PlotterWindow

MAX_BATCH_SIZE = 20000

EPOCHS_MULTIPLIER = 1


class TripletNet(nn.Module):
    def __init__(self, backbone: NoFlatteningBackbone, alpha: float):
        super(TripletNet, self).__init__()
        self.feature_extractor = backbone
        self.alpha = alpha

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
        minibatches = batch.split(split_size=MAX_BATCH_SIZE)
        xs = []
        for minibatch in minibatches:
            output = self.feature_extractor(minibatch)
            output = output.view(output.size(0), -1)
            xs.append(output)
        x = torch.cat(xs)
        return x

    def get_prototypes(self, support_set: torch.Tensor):
        return torch.mean(support_set, dim=1)

    def forward(self, support_set: torch.Tensor, query_set: torch.Tensor) -> torch.Tensor:
        n_classes = support_set.size(0)
        support_set_size = support_set.size(1)
        query_set_size = query_set.size(0)

        support_set_features = self.extract_features(remove_dim(support_set, 1)).view(n_classes,
                                                                                      support_set_size, -1)

        query_set_features = self.extract_features(query_set)

        class_prototypes = self.get_prototypes(support_set_features)

        query_set_features_prepared = query_set_features.unsqueeze(1).repeat_interleave(repeats=n_classes,
                                                                                        dim=1)

        distance = torch.sum((class_prototypes.unsqueeze(0).repeat_interleave(repeats=query_set_size,
                                                                              dim=0) -
                              query_set_features_prepared).pow(2), dim=2)

        return -distance

    def triplet_loss(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor,
                     average=True) -> torch.Tensor:
        anchor = self.extract_features(anchor)
        positive = self.extract_features(positive)
        negative = self.extract_features(negative)

        positive_dist = (anchor - positive).pow(2).sum(1)
        negative_dist = (anchor - negative).pow(2).sum(1)
        diff = F.relu(positive_dist - negative_dist + self.alpha)
        loss = torch.sum(diff) if not average else torch.mean(diff)
        return loss


def train_tripletnet(base_subdataset: LabeledSubdataset, val_subdataset: LabeledSubdataset, n_shot: int, n_way: int,
                     n_iterations: int, batch_size: int, eval_period: int,
                     val_batch_size: int,
                     image_size: int,
                     balanced_batches: bool,
                     alpha=0.5,
                     backbone_name='resnet12-np-o',
                     device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), **kwargs):
    session_info = {
        "task": "few-shot learning",
        "model": "TripletNet",
        "feature_extractor": backbone_name,
        "n_iterations": n_iterations,
        "eval_period": eval_period,
        # "dataset": dataset_name,
        # "optimizer": optimizer_name,
        "batch_size": batch_size,
        "val_batch_size": val_batch_size,
        "n_shot": n_shot,
        "n_way": n_way,
        "alpha": alpha,
        "optimizer": 'adam',
        "image_size": image_size,
        "balanced_batches": balanced_batches,
    }

    session_info.update(kwargs)

    backbone = FEATURE_EXTRACTORS[backbone_name]()
    model = TripletNet(backbone=backbone, alpha=alpha).to(device)

    optimizer = OPTIMIZERS['adam'](model=model)

    base_sampler = TripletBatchSampler(subdataset=base_subdataset, batch_size=batch_size)
    val_sampler = FSLEpisodeSampler(subdataset=val_subdataset, n_way=n_way, n_shot=n_shot, batch_size=val_batch_size,
                                    balanced=balanced_batches)

    loss_plotter = PlotterWindow(interval=1000)
    accuracy_plotter = PlotterWindow(interval=1000)

    loss_plotter.new_line('Loss')
    # accuracy_plotter.new_line('Train Accuracy')
    accuracy_plotter.new_line('Validation Accuracy')

    losses = []
    # acc_train = []
    acc_val = []
    val_iters = []

    best_accuracy = 0
    best_iteration = -1

    print("Training started for parameters:")
    print(session_info)
    print()

    start_time = time.time()

    for iteration in range(n_iterations):
        model.train()
        anchor, positive, negative = base_sampler.sample()

        optimizer.zero_grad()
        loss = model.triplet_loss(anchor, positive, negative)
        loss.backward()
        optimizer.step()

        # labels_pred = output.argmax(dim=1)
        # labels = query_labels
        # cur_accuracy = accuracy(labels=labels, labels_pred=labels_pred)

        loss_plotter.add_point('Loss', iteration, loss.item())
        # accuracy_plotter.add_point('Train Accuracy', iteration, cur_accuracy)

        losses.append(loss.item())
        # acc_train.append(cur_accuracy)

        if iteration % eval_period == 0 or iteration == n_iterations - 1:
            val_start_time = time.time()

            val_accuracy = evaluate_solution_episodes(model, val_sampler)
            accuracy_plotter.add_point('Validation Accuracy', iteration, val_accuracy)

            acc_val.append(val_accuracy)
            val_iters.append(iteration + 1)

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

    session = Session()
    session.build(name="TripletNet", comment=r"Triplet loss",
                  **session_info)
    torch.save(model, os.path.join(session.data['output_dir'], "trained_model_state_dict.tar"))
    iters = list(range(1, n_iterations + 1))

    plt.figure(figsize=(20, 20))
    plt.plot(iters, losses, label="Loss")
    plt.legend()
    plt.savefig(os.path.join(session.data['output_dir'], "loss_plot.png"))

    plt.figure(figsize=(20, 20))
    # plt.plot(iters, acc_train, label="Train Accuracy")
    plt.plot(val_iters, acc_val, label="Test Accuracy")
    plt.legend()
    plt.savefig(os.path.join(session.data['output_dir'], "acc_plot.png"))

    session.save_info()


if __name__ == '__main__':
    torch.random.manual_seed(2002)
    random.seed(2002)

    DATASET_NAME = 'google-landmarks'
    BASE_CLASSES = 4000
    AUGMENT_PROB = 1.0
    ITERATIONS = 10000 * EPOCHS_MULTIPLIER
    N_WAY = 5
    EVAL_PERIOD = 100
    RECORD = 600
    IMAGE_SIZE = 84
    BACKBONE = 'conv64-p-o'
    # BACKBONE = 'resnet18'
    BATCH_SIZE = 64 // EPOCHS_MULTIPLIER
    VAL_BATCH_SIZE = 5 // EPOCHS_MULTIPLIER
    BALANCED_BATCHES = True

    # N_SHOT = 5

    print("Preparations for training...")
    dataset = LABELED_DATASETS[DATASET_NAME](augment_prob=AUGMENT_PROB, image_size=IMAGE_SIZE)
    base_subdataset, val_subdataset = dataset.subdataset.extract_classes(BASE_CLASSES)
    base_subdataset.set_test(False)
    val_subdataset.set_test(True)

    for N_SHOT in (1,):
        train_tripletnet(base_subdataset=base_subdataset, val_subdataset=val_subdataset, n_shot=N_SHOT, n_way=N_WAY,
                         n_iterations=ITERATIONS, batch_size=BATCH_SIZE,
                         eval_period=EVAL_PERIOD,
                         record=RECORD,
                         augment=AUGMENT_PROB,
                         dataset=DATASET_NAME,
                         base_classes=BASE_CLASSES,
                         dataset_classes=dataset.CLASSES,
                         image_size=IMAGE_SIZE,
                         backbone_name=BACKBONE,
                         balanced_batches=BALANCED_BATCHES,
                         val_batch_size=VAL_BATCH_SIZE)
