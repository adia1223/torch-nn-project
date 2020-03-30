import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn

from data import LABELED_DATASETS, LabeledSubdataset
from models.images.classification.few_shot_learning import evaluate_solution, accuracy, FSLEpisodeSampler, \
    FEATURE_EXTRACTORS, ProtoNetBasedFSLSolution
from sessions import Session
from utils import pretty_time
from visualization.plots import PlotterWindow


class Decoder(nn.Module):
    def __init__(self, z_size, ndf=1024, ncf=64):
        super(Decoder, self).__init__()
        self.ndf = ndf
        self.ncf = ncf

        self.dfc3 = nn.Linear(z_size, ndf)
        self.bn3 = nn.BatchNorm1d(ndf)
        self.dfc2 = nn.Linear(ndf, ndf)
        self.bn2 = nn.BatchNorm1d(ndf)
        self.dfc1 = nn.Linear(ndf, ncf * 6 * 6)
        self.bn1 = nn.BatchNorm1d(ncf * 6 * 6)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.dconv5 = nn.ConvTranspose2d(ncf, ncf, 3, padding=0)
        self.dconv4 = nn.ConvTranspose2d(ncf, ncf * 2, 3, padding=1)
        self.dconv3 = nn.ConvTranspose2d(ncf * 2, ncf * 2, 3, padding=1)
        self.dconv2 = nn.ConvTranspose2d(ncf * 2, ncf // 2, 5, padding=2)
        self.dconv1 = nn.ConvTranspose2d(ncf // 2, 3, 12, stride=4, padding=4)

    def forward(self, x):  # ,i1,i2,i3):

        x = self.dfc3(x)
        # x = F.relu(x)
        x = F.relu(self.bn3(x))

        x = self.dfc2(x)
        x = F.relu(self.bn2(x))
        # x = F.relu(x)
        x = self.dfc1(x)
        x = F.relu(self.bn1(x))
        # x = F.relu(x)
        # print(x.size())
        x = x.view(x.size(0), self.ncf, 6, 6)
        # print (x.size())
        x = self.upsample1(x)
        # print x.size()
        x = self.dconv5(x)
        # print x.size()
        x = F.relu(x)
        # print x.size()
        x = F.relu(self.dconv4(x))
        # print x.size()
        x = F.relu(self.dconv3(x))
        # print x.size()
        x = self.upsample1(x)
        # print x.size()
        x = self.dconv2(x)
        # print x.size()
        x = F.relu(x)
        x = self.upsample1(x)
        # print x.size()
        x = self.dconv1(x)
        # print(x.size())
        x = torch.sigmoid(x)
        # print x
        return x


class ProtoNetAE(ProtoNetBasedFSLSolution):
    def __init__(self, backbone: nn.Module):
        super(ProtoNetAE, self).__init__()
        self.feature_extractor = backbone
        self.z_size = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Sequential()

        self.decoder = Decoder(self.z_size)
        # self.loss_fn = nn.CrossEntropyLoss()

    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(batch)

    def get_prototypes(self, support_set: torch.Tensor, query_set: torch.Tensor) -> torch.Tensor:
        return torch.mean(support_set, dim=1)

    def autoencoder(self, batch: torch.Tensor) -> torch.Tensor:
        z = self.feature_extractor(batch)
        return self.decoder(z)


def train_protonet_vae(base_subdataset: LabeledSubdataset, val_subdataset: LabeledSubdataset, n_shot: int, n_way: int,
                       train_n_way: int,
                       n_ae_iterations: int,
                       n_iterations: int, batch_size: int, eval_period: int,
                       backbone_name='resnet18', lr=0.01,
                       device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), **kwargs):
    session_info = {
        "task": "few-shot learning",
        "model": "ProtoNET_AE",
        "feature_extractor": backbone_name,
        "n_iterations": n_iterations,
        "n_ae_iterations": n_ae_iterations,
        "eval_period": eval_period,
        # "dataset": dataset_name,
        # "optimizer": optimizer_name,
        "batch_size": batch_size,
        "n_shot": n_shot,
        "n_way": n_way,
        "train_n_way": train_n_way,
        "optimizer": 'adam'
    }

    session_info.update(kwargs)

    backbone = FEATURE_EXTRACTORS[backbone_name]()
    model = ProtoNetAE(backbone=backbone).to(device)
    print(model)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr
                                 # , nesterov=True, weight_decay=0.0005, momentum=0.9
                                 )

    base_sampler = FSLEpisodeSampler(subdataset=base_subdataset, n_way=train_n_way, n_shot=n_shot,
                                     batch_size=batch_size)
    val_sampler = FSLEpisodeSampler(subdataset=val_subdataset, n_way=n_way, n_shot=n_shot, batch_size=batch_size)

    # ae_data_loader = DataLoader(dataset=base_subdataset, batch_size=batch_size, shuffle=True, num_workers=4)

    proto_loss_fn = nn.CrossEntropyLoss()
    ae_loss_fn = nn.MSELoss()

    loss_plotter = PlotterWindow(interval=1000)
    accuracy_plotter = PlotterWindow(interval=1000)

    loss_plotter.new_line('Loss')
    loss_plotter.new_line('AE Loss')
    loss_plotter.new_line('Proto Loss')
    accuracy_plotter.new_line('Train Accuracy')
    accuracy_plotter.new_line('Validation Accuracy')

    losses = []
    losses_ae = []
    losses_proto = []
    acc_train = []
    acc_val = []
    val_iters = []

    best_accuracy = 0
    best_iteration = -1

    print("Training started for parameters:")
    print(session_info)
    print()

    start_time = time.time()

    print("Autoencoder stage")
    for iteration in range(n_ae_iterations):
        model.train()
        support_set, batch = base_sampler.sample()
        query_set, query_labels = batch
        query_set = query_set.to(device)

        optimizer.zero_grad()
        autoencoded = model.autoencoder(query_set)
        loss_ae = ae_loss_fn(autoencoded, query_set)
        loss_ae.backward()
        optimizer.step()

        loss_plotter.add_point('Loss', iteration, -1)
        loss_plotter.add_point('AE Loss', iteration, loss_ae.item())
        loss_plotter.add_point('Proto Loss', iteration, -1)
        accuracy_plotter.add_point('Train Accuracy', iteration, -1)

        losses.append(-1)
        losses_ae.append(loss_ae.item())
        losses_proto.append(-1)
        acc_train.append(-1)

        if iteration % eval_period == 0 or iteration == n_ae_iterations - 1:
            cur_time = time.time()
            time_used = cur_time - start_time
            time_per_iteration = time_used / (iteration + 1)

            print()
            print("[%d/%d] = %.2f%%\t\tLoss: %.4f" % (
                iteration + 1, n_ae_iterations, (iteration + 1) / n_ae_iterations * 100, loss_ae.item()))

            print('Average iteration time: %s\tEstimated execution time: %s' % (
                pretty_time(time_per_iteration),
                pretty_time(time_per_iteration * (n_ae_iterations - iteration - 1)),
            ))
            print()

    print("Combined stage")

    for iteration in range(n_iterations):
        model.train()

        support_set, batch = base_sampler.sample()
        query_set, query_labels = batch
        query_set = query_set.to(device)
        query_labels = query_labels.to(device)

        optimizer.zero_grad()
        # output, loss = model.forward_with_loss(support_set, query_set, query_labels)
        output = model(support_set, query_set)
        loss_proto = proto_loss_fn(output, query_labels) * 0.1
        autoencoded = model.autoencoder(query_set)
        loss_ae = ae_loss_fn(autoencoded, query_set)
        loss = loss_proto + loss_ae
        loss.backward()
        optimizer.step()

        labels_pred = output.argmax(dim=1)
        labels = query_labels
        cur_accuracy = accuracy(labels=labels, labels_pred=labels_pred)

        loss_plotter.add_point('Loss', iteration + n_ae_iterations, loss.item())
        loss_plotter.add_point('AE Loss', iteration + n_ae_iterations, loss_ae.item())
        loss_plotter.add_point('Proto Loss', iteration + n_ae_iterations, loss_proto.item())
        accuracy_plotter.add_point('Train Accuracy', iteration + n_ae_iterations, cur_accuracy)

        losses.append(loss.item())
        losses_ae.append(loss_ae.item())
        losses_proto.append(loss_proto.item())
        acc_train.append(cur_accuracy)

        if iteration % eval_period == 0 or iteration == n_iterations - 1:
            val_start_time = time.time()

            val_accuracy = evaluate_solution(model, val_sampler)
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
    session.build(name="FSL_PROTONET_AE", comment=r"Few-Shot Learning solution based on autoencoder",
                  **session_info)

    torch.save(model, os.path.join(session.data['output_dir'], "trained_model_state_dict.tar"))
    iters = list(range(1, n_ae_iterations + n_iterations + 1))

    plt.figure(figsize=(20, 20))
    plt.plot(iters, losses, label="Loss")
    plt.plot(iters, losses_ae, label="AE Loss")
    plt.plot(iters, losses_proto, label="Proto Loss")
    plt.legend()
    plt.savefig(os.path.join(session.data['output_dir'], "loss_plot.png"))

    plt.figure(figsize=(20, 20))
    plt.plot(iters, acc_train, label="Train Accuracy")
    plt.plot(val_iters, acc_val, label="Test Accuracy")
    plt.legend()
    plt.savefig(os.path.join(session.data['output_dir'], "acc_plot.png"))

    session.save_info()


if __name__ == '__main__':

    DATASET_NAME = 'cub'
    BASE_CLASSES = 80
    AUGMENT_PROB = 1.0
    AE_ITERATIONS = 20000
    # AE_ITERATIONS = 20
    ITERATIONS = 20000
    # ITERATIONS = 20
    BATCH_SIZE = 4
    N_WAY = 5
    TRAIN_N_WAY = 15
    EVAL_PERIOD = 1000
    RECORD = -1

    # N_SHOT = 5

    print("Preparations for training...")
    dataset = LABELED_DATASETS[DATASET_NAME](augment_prob=AUGMENT_PROB)
    base_subdataset, val_subdataset = dataset.subdataset.extract_classes(BASE_CLASSES)
    base_subdataset.set_test(False)
    val_subdataset.set_test(True)

    for N_SHOT in (5, 1):
        train_protonet_vae(base_subdataset=base_subdataset, val_subdataset=val_subdataset, n_shot=N_SHOT, n_way=N_WAY,
                           n_iterations=ITERATIONS,
                           n_ae_iterations=AE_ITERATIONS,
                           batch_size=BATCH_SIZE,
                           eval_period=EVAL_PERIOD,
                           record=RECORD,
                           augment=AUGMENT_PROB,
                           dataset=DATASET_NAME,
                           base_classes=BASE_CLASSES,
                           train_n_way=TRAIN_N_WAY)
