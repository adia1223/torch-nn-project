import time

import torch
from torch import nn

from data import LabeledSubdataset, LABELED_DATASETS
from models.images.classification.few_shot_learning import ProtoNetBasedFSLSolution, FSLEpisodeSampler, \
    FEATURE_EXTRACTORS, OPTIMIZERS, accuracy, evaluate_solution
from utils import pretty_time
from visualization.plots import PlotterWindow


class ProtoNet(ProtoNetBasedFSLSolution):
    def __init__(self, backbone: nn.Module):
        super(ProtoNet, self).__init__()
        self.feature_extractor = backbone
        self.feature_extractor.fc = nn.Sequential()

    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(batch)

    def get_prototypes(self, support_set: torch.Tensor, query_set: torch.Tensor) -> torch.Tensor:
        return torch.mean(support_set, dim=1)


def train_model(base_subdataset: LabeledSubdataset, val_subdataset: LabeledSubdataset, n_shot: int, n_way: int,
                n_iterations: int, batch_size: int, eval_period: int,
                backbone_name='resnet18',
                optimizer_name='adam',
                device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), session_info=None):
    if session_info is None:
        session_info = dict()
    backbone = FEATURE_EXTRACTORS[backbone_name]()
    model = ProtoNet(backbone=backbone).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = OPTIMIZERS[optimizer_name](model=model)

    base_sampler = FSLEpisodeSampler(subdataset=base_subdataset, n_way=15, n_shot=n_shot, batch_size=batch_size)
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
        output = model.forward(support_set, query_set)
        loss = loss_fn(output, query_labels)
        loss.backward()
        optimizer.step()

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
    dataset = LABELED_DATASETS['miniImageNet'](augment_prob=0.0)
    base_subdataset, val_subdataset = dataset.subdataset.extract_classes(20)
    base_subdataset.set_test(False)
    val_subdataset.set_test(True)
    train_model(base_subdataset, val_subdataset, 5, 5, 5000, 16, 500)
