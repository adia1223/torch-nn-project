import time

import torch
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from torch import device as torch_device
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from sessions import Session
from training import pretty_time
from visualization.plots import PlotterWindow


def accuracy(labels, labels_pred):
    return accuracy_score(labels_pred.cpu(), labels.cpu())


def balanced_accuracy(labels, labels_pred):
    return balanced_accuracy_score(labels_pred.cpu(), labels.cpu())


def eval_model(model: nn.Module, dataloader: DataLoader, loss: nn.Module, device: torch_device):
    with torch.no_grad():
        model.eval()

        ac_score = 0
        ba_score = 0
        eval_loss = 0

        for data in dataloader:
            x, labels, _ = data
            x = x.to(device)
            labels = labels.to(device).long()

            y_pred = model(x)
            labels_pred = y_pred.argmax(dim=1)

            ac_score += accuracy(labels_pred, labels)
            ba_score += balanced_accuracy(labels_pred, labels)
            eval_loss += loss(y_pred, labels).item()

        ac_score /= len(dataloader)
        ba_score /= len(dataloader)
        eval_loss /= len(dataloader)
        return eval_loss, ac_score, ba_score,


def train_classifier(session: Session,
                     dataloader: DataLoader,
                     val_dataloader: DataLoader,
                     model: nn.Module,
                     optimizer: Optimizer,
                     loss: nn.Module,
                     n_classes: int,
                     epochs: int,
                     cur_epoch: int = 0,
                     device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                     verbosity_period=50,
                     eval_epoch_period=1,
                     ):
    model = model.to(device)
    loss = loss.to(device)

    loss_plotter = PlotterWindow(interval=1000)
    eval_plotter = PlotterWindow(interval=1000)

    it_per_epoch = len(dataloader)

    session.checkpoint()
    print("Training started\n")

    start_time = time.time()

    cur_it = it_per_epoch * cur_epoch
    total_it = it_per_epoch * epochs

    loss_plotter.new_line('Loss')
    losses = []

    loss_plotter.new_line('Val Loss')
    val_losses = []

    eval_plotter.new_line('Accuracy')
    acc_scores = []
    eval_plotter.new_line('Balanced Accuracy')
    ba_scores = []

    eval_iters = [0]

    val_loss, acc, ba_score = eval_model(model, val_dataloader, loss, device)
    val_losses.append(val_loss)
    loss_plotter.add_point("Val Loss", cur_it, val_loss)
    acc_scores.append(acc)
    eval_plotter.add_point('Accuracy', cur_it, acc)
    ba_scores.append(ba_score)
    eval_plotter.add_point('Balanced Accuracy', cur_it, ba_score)

    for epoch in range(cur_epoch, epochs):
        model.train()
        for epoch_it, data in enumerate(dataloader, 0):
            x, y, _ = data
            x = x.to(device)
            y = y.to(device).long()

            optimizer.zero_grad()

            y_pred = model(x)
            model_loss = loss(y_pred, y)
            model_loss.backward()
            optimizer.step()

            cur_it += 1

            cur_time = time.time()
            delta_time = cur_time - start_time
            time_per_it = delta_time / cur_it

            loss_plotter.add_point('Loss', cur_it, model_loss.item())
            losses.append(model_loss.item())

            if epoch_it % verbosity_period == 0 or epoch_it == len(dataloader) - 1:
                print('[%d/%d][%d/%d] = %.2f%%\t\tLoss: %.4f' %
                      (epoch + 1, epochs, epoch_it, it_per_epoch, cur_it / total_it * 100,
                       model_loss.item()))
                print('Average iteration time: %s\tAverage epoch time: %s\tEstimated execution time: %s' % (
                    pretty_time(time_per_it),
                    pretty_time(time_per_it * it_per_epoch),
                    pretty_time(time_per_it * (it_per_epoch * epochs - cur_it)),
                ))
                print()

        if val_dataloader is not None and (epoch % eval_epoch_period == 0 or epoch == epochs - 1):
            eval_iters.append(cur_it)

            val_start_time = time.time()

            val_loss, acc, ba_score = eval_model(model, val_dataloader, loss, device)
            val_losses.append(val_loss)
            loss_plotter.add_point("Val Loss", cur_it, val_loss)

            acc_scores.append(acc)
            eval_plotter.add_point('Accuracy', cur_it, acc)
            ba_scores.append(ba_score)
            eval_plotter.add_point('Balanced Accuracy', cur_it, ba_score)
            val_end_time = time.time()
            val_time = val_end_time - val_start_time
            print("[%d/%d] = %.2f%%\tValidation:\taccuracy = %.4f\tValidation AP = %.4f" % (
                epoch + 1, epochs, cur_it / total_it * 100, acc, ba_score))
            print('Validation time: %s' % (pretty_time(val_time),))
            print()

            session.checkpoint()

    session.info['test_accuracy'] = acc_scores[-1]
    session.info['test_balanced_accuracy'] = ba_scores[-1]

    cur_time = time.time()
    delta_time = cur_time - start_time
    print("Training finished. Total execution time: %s" % pretty_time(delta_time))
    print()

    session.info['execution_time'] = delta_time

    return total_it, losses, val_losses, eval_iters, acc_scores, ba_scores
