import os

import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torch import nn, optim

from data.cifar10 import CIFAR10Dataset
from data.gtsrb import GTSRBDataset
from models.images.classification.transfer_learning import ResNet18Classifier, GoogLeNetClassifier, \
    ResNet18CosineClassifier
from sessions import Session
from training import simple


class ClassifierSession(Session):
    def __init__(self,
                 dataset,
                 n_classes,
                 model,
                 loss,
                 optimizer,
                 state_file,
                 name,
                 comment,
                 epochs,
                 batch_size,
                 val_batch_size,
                 dataloader_workers,
                 eval_period,
                 **kwargs
                 ):
        self.dataset = dataset
        self.n_classes = n_classes

        self.eval_period = eval_period

        self.train_dataset = self.dataset.train()
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size,
                                                            shuffle=True, num_workers=dataloader_workers)

        self.test_dataset = self.dataset.test()
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=val_batch_size,
                                                           shuffle=True, num_workers=dataloader_workers)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = model

        self.loss = loss

        self.optimizer = optimizer

        self.build(
            state_file=state_file,
            n_classes=n_classes,
            name=name,
            comment=comment,
            epochs=epochs,
            batch_size=batch_size,
            dataloader_workers=dataloader_workers,
            **kwargs
        )

    def __restore__(self, data_file):
        super(ClassifierSession, self).__restore__(data_file)
        torch_state_file = os.path.join(self.data['checkpoint_dir'], 'torch_state.tar')
        checkpoint = torch.load(torch_state_file)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss.load_state_dict(checkpoint['loss_state_dict'])

    def checkpoint(self):
        super(ClassifierSession, self).checkpoint()
        torch_state_file = os.path.join(self.data['checkpoint_dir'], 'torch_state.tar')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_state_dict': self.loss.state_dict(),
        }, torch_state_file)

    def training(self):
        return simple.train_classifier(self,
                                       model=self.model,
                                       dataloader=self.train_dataloader,
                                       val_dataloader=self.test_dataloader,
                                       optimizer=self.optimizer,
                                       loss=self.loss,
                                       epochs=self.data['epochs'],
                                       cur_epoch=0,
                                       n_classes=self.n_classes,
                                       eval_epoch_period=self.eval_period
                                       )

    def save_output(self, total_it, losses, val_losses, eval_iters, acc_scores, ba_scores):
        torch.save(self.model, os.path.join(self.data['output_dir'], "trained_model_state_dict.tar"))
        iters = list(range(1, total_it + 1))

        plt.figure(figsize=(20, 20))
        plt.plot(iters, losses, label="Cross Entropy Loss")
        plt.plot(eval_iters, val_losses, label="Cross Entropy Loss On test")
        plt.legend()
        plt.savefig(os.path.join(self.data['output_dir'], "loss_plot.png"))

        plt.figure(figsize=(20, 20))
        plt.plot(eval_iters, acc_scores, label="Accuracy")
        plt.plot(eval_iters, ba_scores, label="Balanced Accuracy")
        plt.legend()
        plt.savefig(os.path.join(self.data['output_dir'], "eval_plot.png"))

    def run(self):
        self.save_output(
            *self.training()
        )


class ResNet18GTSRB(ClassifierSession):
    def __init__(self,
                 name,
                 comment,
                 freeze_backbone,
                 learning_rate,
                 epochs,
                 batch_size,
                 val_batch_size,
                 dataloader_workers,
                 eval_period,
                 augment_prob,
                 reduce,
                 pretrained,
                 state_file=None,
                 ):
        dataset = GTSRBDataset(reduce=reduce, augment_prob=augment_prob)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = ResNet18Classifier(dataset.CLASSES, freeze_backbone=freeze_backbone, pretrained=pretrained)

        loss = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        super(ResNet18GTSRB, self).__init__(
            dataset=dataset,
            model=model,
            loss=loss,
            optimizer=optimizer,
            state_file=state_file,
            name=name,
            comment=comment,
            freeze_backbone=freeze_backbone,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            dataloader_workers=dataloader_workers,
            n_classes=dataset.CLASSES,
            reduce=reduce,
            augment_prob=augment_prob,
            eval_period=eval_period,
            pretrained=pretrained,
        )


class ResNet18CIFAR10(ClassifierSession):
    def __init__(self,
                 name,
                 comment,
                 freeze_backbone,
                 learning_rate,
                 epochs,
                 batch_size,
                 val_batch_size,
                 dataloader_workers,
                 eval_period,
                 augment_prob,
                 reduce,
                 pretrained,
                 state_file=None,
                 ):
        dataset = CIFAR10Dataset(reduce=reduce, augment_prob=augment_prob)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = ResNet18Classifier(dataset.CLASSES, freeze_backbone=freeze_backbone, pretrained=pretrained)

        loss = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        super(ResNet18CIFAR10, self).__init__(
            dataset=dataset,
            model=model,
            loss=loss,
            optimizer=optimizer,
            state_file=state_file,
            name=name,
            comment=comment,
            freeze_backbone=freeze_backbone,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            dataloader_workers=dataloader_workers,
            n_classes=dataset.CLASSES,
            reduce=reduce,
            augment_prob=augment_prob,
            eval_period=eval_period,
            pretrained=pretrained,
        )


class GoogLeNetGTSRB(ClassifierSession):
    def __init__(self,
                 name,
                 comment,
                 freeze_backbone,
                 learning_rate,
                 epochs,
                 batch_size,
                 val_batch_size,
                 dataloader_workers,
                 eval_period,
                 reduce,
                 augment_prob,
                 pretrained,
                 state_file=None,
                 ):
        dataset = GTSRBDataset(reduce=reduce, augment_prob=augment_prob)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = GoogLeNetClassifier(dataset.CLASSES, freeze_backbone=freeze_backbone, pretrained=pretrained)

        loss = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        super(GoogLeNetGTSRB, self).__init__(
            dataset=dataset,
            model=model,
            loss=loss,
            optimizer=optimizer,
            state_file=state_file,
            name=name,
            comment=comment,
            freeze_backbone=freeze_backbone,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            dataloader_workers=dataloader_workers,
            n_classes=dataset.CLASSES,
            reduce=reduce,
            augment_prob=augment_prob,
            eval_period=eval_period,
            pretrained=pretrained,
        )


class ResNet18CosineGTSRB(ClassifierSession):
    def __init__(self,
                 name,
                 comment,
                 freeze_backbone,
                 learning_rate,
                 epochs,
                 batch_size,
                 val_batch_size,
                 dataloader_workers,
                 eval_period,
                 augment_prob,
                 reduce,
                 pretrained,
                 state_file=None,
                 ):
        dataset = GTSRBDataset(reduce=reduce, augment_prob=augment_prob)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = ResNet18CosineClassifier(dataset.CLASSES, freeze_backbone=freeze_backbone, pretrained=pretrained)

        loss = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        super(ResNet18CosineGTSRB, self).__init__(
            dataset=dataset,
            model=model,
            loss=loss,
            optimizer=optimizer,
            state_file=state_file,
            name=name,
            comment=comment,
            freeze_backbone=freeze_backbone,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            dataloader_workers=dataloader_workers,
            n_classes=dataset.CLASSES,
            reduce=reduce,
            augment_prob=augment_prob,
            eval_period=eval_period,
            pretrained=pretrained,
        )


class ResNet18CosineCIFAR10(ClassifierSession):
    def __init__(self,
                 name,
                 comment,
                 freeze_backbone,
                 learning_rate,
                 epochs,
                 batch_size,
                 val_batch_size,
                 dataloader_workers,
                 eval_period,
                 augment_prob,
                 reduce,
                 pretrained,
                 state_file=None,
                 ):
        dataset = CIFAR10Dataset(reduce=reduce, augment_prob=augment_prob)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = ResNet18CosineClassifier(dataset.CLASSES, freeze_backbone=freeze_backbone, pretrained=pretrained)

        loss = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        super(ResNet18CosineCIFAR10, self).__init__(
            dataset=dataset,
            model=model,
            loss=loss,
            optimizer=optimizer,
            state_file=state_file,
            name=name,
            comment=comment,
            freeze_backbone=freeze_backbone,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            dataloader_workers=dataloader_workers,
            n_classes=dataset.CLASSES,
            reduce=reduce,
            augment_prob=augment_prob,
            eval_period=eval_period,
            pretrained=pretrained,
        )
