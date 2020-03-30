import time

import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import models

from data import LABELED_DATASETS, LabeledSubdataset
from history.index import save_record
from models.images.classification.backbones import ResNet18NoPooling
from models.images.classification.meta_learning_few_shot import MODELS, FewShotLearningTask, BaselineClassifier, \
    SupportSetMeanFeaturesModel
from training import pretty_time
from visualization.plots import PlotterWindow


def accuracy(labels, labels_pred):
    return accuracy_score(labels_pred.cpu(), labels.cpu())


def balanced_accuracy(labels, labels_pred):
    return balanced_accuracy_score(labels_pred.cpu(), labels.cpu())


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
    'resnet18-np': lambda: ResNet18NoPooling(pretrained=False),
}


class EpisodeSampler(Dataset):
    def __init__(self, subdataset: LabeledSubdataset, n_way: int, n_shot: int, batch_size: int,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.subdataset = subdataset
        self.n_way = n_way
        self.n_shot = n_shot
        self.batch_size = batch_size
        self.device = device

    def sample(self):
        cur_subdataset, _ = self.subdataset.extract_classes(self.n_way)
        support_subdataset = cur_subdataset.balance(self.n_shot)
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

        batch = cur_subdataset.random_batch(self.batch_size)
        for i in range(len(batch[1])):
            batch[1][i] = classes_mapping[batch[1][i].item()]

        return support_set, batch


def evaluate(model: FewShotLearningTask, sampler: EpisodeSampler, n_iterations: int = 600, no_grad=True):
    device = sampler.device
    accuracy_sum = 0
    if no_grad:
        with torch.no_grad():
            model.eval()
            for iteration in range(n_iterations):
                support_set, batch = sampler.sample()
                x, y = batch
                x = x.to(device)
                y = y.to(device)

                loss, y_pred = model.test_step(x, y, support_set)
                labels_pred = y_pred.argmax(dim=1)
                labels = y
                cur_accuracy = accuracy(labels=labels, labels_pred=labels_pred)
                accuracy_sum += cur_accuracy
    else:
        for iteration in range(n_iterations):
            support_set, batch = sampler.sample()
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            loss, y_pred = model.test_step(x, y, support_set)
            labels_pred = y_pred.argmax(dim=1)
            labels = y
            cur_accuracy = accuracy(labels=labels, labels_pred=labels_pred)
            accuracy_sum += cur_accuracy

    return accuracy_sum / n_iterations


def meta_learning_train(model: SupportSetMeanFeaturesModel, optimizer, base_subdataset, val_subdataset, n_iterations,
                        n_shot, n_way, batch_size, eval_period, device, session_info):
    base_sampler = EpisodeSampler(subdataset=base_subdataset, n_way=n_way, n_shot=n_shot, batch_size=batch_size)
    val_sampler = EpisodeSampler(subdataset=val_subdataset, n_way=n_way, n_shot=n_shot, batch_size=batch_size)

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
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        loss, y_pred = model.train_step(x, y, support_set, optimizer)
        labels_pred = y_pred.argmax(dim=1)
        labels = y
        cur_accuracy = accuracy(labels=labels, labels_pred=labels_pred)

        loss_plotter.add_point('Loss', iteration, loss.item())
        accuracy_plotter.add_point('Train Accuracy', iteration, cur_accuracy)

        if iteration % eval_period == 0 or iteration == n_iterations - 1:
            val_start_time = time.time()

            val_accuracy = evaluate(model, val_sampler)
            accuracy_plotter.add_point('Validation Accuracy', iteration, val_accuracy)

            cur_time = time.time()

            val_time = cur_time - val_start_time
            time_used = cur_time - start_time
            time_per_iteration = time_used / (iteration + 1)

            print("[%d/%d] = %.2f%%\t\tLoss: %.4f" % (
                iteration + 1, n_iterations, (iteration + 1) / n_iterations * 100, loss.item()))
            print("Train accuracy: %.3f\tValidation accuracy: %.3f" % (cur_accuracy, val_accuracy,))

            if val_accuracy > best_accuracy:
                print("Best validation accuracy!")
                best_accuracy = val_accuracy
                best_iteration = iteration

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


def baseline_train(model: BaselineClassifier, optimizer, base_subdataset: LabeledSubdataset,
                   val_subdataset: LabeledSubdataset, n_iterations,
                   n_shot, n_way, batch_size, eval_period, device, session_info, train_batch_size=16):
    dataloader = DataLoader(dataset=base_subdataset, batch_size=train_batch_size, shuffle=True, num_workers=4)
    val_sampler = EpisodeSampler(subdataset=val_subdataset, n_way=n_way, n_shot=n_shot, batch_size=batch_size)

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

    loss = None

    clf = model.classifier(model.backbone_features, base_subdataset.base_dataset.classes).to(device)

    start_time = time.time()
    for iteration in range(n_iterations):
        model.train()
        model.unfreeze_backbone()
        if model.backbone.fc is not clf:
            model.backbone.fc = clf

        cur_accuracy = 0
        cur_loss = 0

        for batch_num, data in enumerate(dataloader):
            x, y, _ = data
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = model.loss(y_pred, y)
            loss.backward()
            optimizer.step()

            labels_pred = y_pred.argmax(dim=1)
            labels = y
            cur_accuracy += accuracy(labels=labels, labels_pred=labels_pred)
            cur_loss += loss.item()

        cur_accuracy /= len(dataloader)
        cur_loss /= len(dataloader)
        loss_plotter.add_point('Loss', iteration, cur_loss)
        accuracy_plotter.add_point('Train Accuracy', iteration, cur_accuracy)

        if iteration % eval_period == 0 or iteration == n_iterations - 1:
            val_start_time = time.time()

            model.freeze_backbone()
            val_accuracy = evaluate(model, val_sampler, no_grad=False)

            accuracy_plotter.add_point('Validation Accuracy', iteration, val_accuracy)

            cur_time = time.time()

            val_time = cur_time - val_start_time
            time_used = cur_time - start_time
            time_per_iteration = time_used / (iteration + 1)

            print("[%d/%d] = %.2f%%\t\tLoss: %.4f" % (
                iteration + 1, n_iterations, (iteration + 1) / n_iterations * 100, loss.item()))
            print("Train accuracy: %.3f\tValidation accuracy: %.3f" % (cur_accuracy, val_accuracy,))

            if val_accuracy > best_accuracy:
                print("Best validation accuracy!")
                best_accuracy = val_accuracy
                best_iteration = iteration

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


def train_model(model_name: str, dataset_name: str, n_shot: int, n_way: int, base_classes: int, n_iterations: int,
                feature_extractor_name='resnet18', augment_prob=0.0,
                optimizer_name: str = 'adam', batch_size: int = 16, eval_period: int = 600,
                device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                **kwargs
                ):
    if model_name in ('RelationNet',):
        feature_extractor_name += '-np'

    session_info = {
        "task": "few-shot learning",
        "model": model_name,
        "feature_extractor": feature_extractor_name,
        "n_iterations": n_iterations,
        "dataset": dataset_name,
        "optimizer": optimizer_name,
        "batch_size": batch_size,
        "n_shot": n_shot,
        "n_way": n_way,
        "augment_prob": augment_prob,
    }

    session_info.update(kwargs)

    dataset = LABELED_DATASETS[dataset_name](augment_prob=augment_prob)
    backbone = FEATURE_EXTRACTORS[feature_extractor_name]()
    model = MODELS[model_name](backbone=backbone).to(device)
    optimizer = OPTIMIZERS[optimizer_name](model=model)

    base_subdataset, val_subdataset = dataset.subdataset.extract_classes(base_classes)
    base_subdataset.set_test(False)
    val_subdataset.set_test(True)

    if model_name in ('Baseline', 'Baseline++'):
        baseline_train(model, optimizer, base_subdataset, val_subdataset, n_iterations, n_shot, n_way, batch_size,
                       eval_period, device, session_info)
    else:
        meta_learning_train(model, optimizer, base_subdataset, val_subdataset, n_iterations, n_shot, n_way, batch_size,
                            eval_period, device, session_info)

    save_record(name="Few-Shot Learning Training", **session_info)


if __name__ == '__main__':
    N_WAY = 5
    RECORD = 35

    DATASET = 'miniImageNet'
    BASE_CLASSES = 80

    # EVAL_PERIOD = 10
    # for cur_model in (
    #         "Baseline",
    #         "Baseline++",
    # ):
    #     for cur_n_shot, cur_n_it in ((5, 100), (1, 100)):
    #         train_model(model_name=cur_model, dataset_name=DATASET, n_shot=cur_n_shot, n_way=N_WAY,
    #                     base_classes=BASE_CLASSES,
    #                     n_iterations=cur_n_it,
    #                     eval_period=EVAL_PERIOD, record=RECORD, augment_prob=1)

    EVAL_PERIOD = 1000
    for cur_model in (
            "ProtoNet",
            # "RelationNet",
    ):
        for cur_n_shot, cur_n_it in ((5, 40000), (1, 40000)):
            train_model(model_name=cur_model, dataset_name=DATASET, n_shot=cur_n_shot, n_way=N_WAY,
                        base_classes=BASE_CLASSES,
                        n_iterations=cur_n_it,
                        eval_period=EVAL_PERIOD, record=RECORD, augment_prob=1)
