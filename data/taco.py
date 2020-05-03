import os
import random
import shutil

import torch
import torchvision
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

import data

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def tensor_loader(path):
    return torch.load(path)


class ImageItem(object):
    def __init__(self, source: ImageFolder, index: int):
        self.source = source
        self.index = index

    def load(self):
        return self.source[self.index][0]


class TacoDataset(data.LabeledDataset):
    CLASSES = 38

    def __init__(self, root=r"C:\datasets\taco\tensors", augment_prob=0.0, reduce=0.0, image_size=84,
                 tensors=True,
                 random_seed=42, **kwargs):
        self.reduce = reduce
        self.tensors = tensors
        random.seed(random_seed)

        resize_train = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
            ]
        )
        resize_test = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
            ]
        )

        augment = transforms.Compose(
            [
                # transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.ColorJitter(),
                # transforms.RandomPerspective(p=0.2, distortion_scale=0.25),
            ]
        )

        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD)
            ]
        )

        self.test_transform = transforms.Compose(
            [
                resize_test,
                normalize
            ]
        )
        self.train_transform = transforms.Compose(
            [
                resize_train,
                transforms.RandomApply([augment], p=augment_prob),
                normalize
            ]
        )
        if not tensors:
            self.source_dataset_train = torchvision.datasets.ImageFolder(root=root)
        else:
            self.source_dataset_train = torchvision.datasets.DatasetFolder(root=root, loader=tensor_loader,
                                                                           extensions=('pt',))

        self.dataset_train_size = len(self.source_dataset_train)
        items = []
        labels = []
        for i in range(self.dataset_train_size):
            items.append(ImageItem(self.source_dataset_train, i))
            labels.append(self.source_dataset_train[i][1])
        is_test = [0] * self.dataset_train_size

        super(TacoDataset, self).__init__(items, labels, is_test)

        self.train_subdataset, self.test_subdataset = self.subdataset.train_test_split()

        if reduce < 1:
            self.train_subdataset = self.train_subdataset.downscale(1 - reduce)
        else:
            self.train_subdataset = self.train_subdataset.balance(reduce)

    def __getitem__(self, item):
        image, label, is_test = super(TacoDataset, self).__getitem__(item)
        if self.tensors:
            return image, label, is_test

        if is_test:
            image = self.test_transform(image)
        else:
            image = self.train_transform(image)

        return image, label, is_test

    def label_stat(self):
        pass

    def train(self):
        return self.train_subdataset

    def test(self):
        return self.test_subdataset


def save_as_tensors(source=r'C:\datasets\taco\images', target=r'C:\datasets\taco\tensors',
                    image_size=84):
    resize = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
    )

    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ]
    )

    transform = transforms.Compose(
        [
            resize,
            normalize
        ]
    )

    os.makedirs(target, exist_ok=True)
    for i, class_label in enumerate(os.listdir(source)):
        cur_source = os.path.join(source, class_label)
        cur_target = os.path.join(target, class_label)
        os.makedirs(cur_target, exist_ok=True)
        for image in os.listdir(cur_source):
            source_image_file = os.path.join(cur_source, image)
            target_file = os.path.join(cur_target, image.replace('.jpg', '.pt'))
            with open(source_image_file, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
                image_tensor = transform(img)
                torch.save(image_tensor, target_file)
        print(class_label, i)


def remove_small(threshold, root=r'C:\datasets\taco\tensors'):
    for label in os.listdir(root):
        path = os.path.join(root, label)
        cnt = len(os.listdir(path))
        if cnt < threshold:
            print(label, cnt)
            shutil.rmtree(path)
            print("Deleted!")

# if __name__ == '__main__':
#     # save_as_tensors()
#     remove_small(10)
