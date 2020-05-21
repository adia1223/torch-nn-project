import os
import random
import shutil
import time
from io import BytesIO
from typing import Union

import requests
import torch
import torchvision
from PIL import Image
from torchvision.datasets import ImageFolder, DatasetFolder
from torchvision.transforms import transforms

import data

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def tensor_loader(path):
    return torch.load(path)


class ImageItem(object):
    def __init__(self, source: Union[ImageFolder, DatasetFolder], index: int):
        self.source = source
        self.index = index

    def load(self):
        return self.source[self.index][0]


class GoogleLandmarksDatasetBase(data.LabeledDataset):
    def __init__(self, root, reduce,
                 random_seed, **kwargs):
        self.CLASSES = len(os.listdir(root))

        self.reduce = reduce
        random.seed(random_seed)

        self.source_dataset_train = torchvision.datasets.DatasetFolder(root=root, loader=tensor_loader,
                                                                       extensions=('pt',))

        self.dataset_train_size = len(self.source_dataset_train)
        items = []
        labels = []
        is_test = [0] * self.dataset_train_size
        for i in range(self.dataset_train_size):
            items.append(ImageItem(self.source_dataset_train, i))
            labels.append(self.source_dataset_train[i][1])

        super(GoogleLandmarksDatasetBase, self).__init__(items, labels, is_test)

        self.train_subdataset, self.test_subdataset = self.subdataset.train_test_split()

        if reduce < 1:
            self.train_subdataset = self.train_subdataset.downscale(1 - reduce)
        else:
            self.train_subdataset = self.train_subdataset.balance(reduce)

    def __getitem__(self, item):
        image, label, is_test = super(GoogleLandmarksDatasetBase, self).__getitem__(item)
        return image, label, is_test

    def label_stat(self):
        pass

    def train(self):
        return self.train_subdataset

    def test(self):
        return self.test_subdataset


class GoogleLandmarksDataset(GoogleLandmarksDatasetBase):
    def __init__(self, root=r"C:\datasets\google-landmarks\train\image-tensors", reduce=0.0,
                 random_seed=42, **kwargs):
        super(GoogleLandmarksDataset, self).__init__(root, reduce, random_seed, **kwargs)


class GoogleLandmarksDataset2(GoogleLandmarksDatasetBase):
    def __init__(self, root=r"C:\datasets\google-landmarks\train\image-tensors-2", reduce=0.0,
                 random_seed=42, **kwargs):
        super(GoogleLandmarksDataset2, self).__init__(root, reduce, random_seed, **kwargs)


class GoogleLandmarksDatasetSelfSupervision(GoogleLandmarksDatasetBase):
    def __init__(self, root=r"C:\datasets\google-landmarks\train\image-tensors-selfsupervision", reduce=0.0,
                 random_seed=42, **kwargs):
        super(GoogleLandmarksDatasetSelfSupervision, self).__init__(root, reduce, random_seed, **kwargs)


class GoogleLandmarksDatasetTest(GoogleLandmarksDatasetBase):
    def __init__(self, root=r"C:\datasets\google-landmarks\test\image-tensors", reduce=0.0,
                 random_seed=42, **kwargs):
        super(GoogleLandmarksDatasetTest, self).__init__(root, reduce, random_seed, **kwargs)


import pandas as pd

ATTEMPTS = 3
SLEEP = 1


def load_from_index(source=r'C:\datasets\google-landmarks\train\filtered_train_2.csv',
                    target=r'C:\datasets\google-landmarks\train\image-tensors-2',
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
    index = pd.read_csv(source)
    for i, row in enumerate(index.iterrows()):
        done = False
        url = None
        attempts = 0
        while not done:
            attempts += 1
            try:
                label = str(row[1]['landmark_id'])
                label_folder = os.path.join(target, label)
                image_id = str(row[1]['id'])
                image_path = os.path.join(label_folder, image_id + r'.pt')
                if os.path.exists(image_path):
                    break

                url = str(row[1]['url'])

                os.makedirs(label_folder, exist_ok=True)

                response = requests.get(url)
                pil_image = Image.open(BytesIO(response.content))
                pil_image = pil_image.convert('RGB')
                image_tensor = transform(pil_image)

                torch.save(image_tensor, image_path)

                done = True
            except (OSError, AttributeError):
                if attempts <= ATTEMPTS:
                    print("Error with url %s, delay for %d second(s)" % (url, SLEEP))
                    time.sleep(SLEEP)
                else:
                    done = True
                    print("Skipped")

        if i % 50 == 0:
            print(i)


CROPS_NUM = 8


def load_from_selfsupervision_index(source=r'C:\datasets\google-landmarks\train\filtered_train_selfsupervision.csv',
                                    target=r'C:\datasets\google-landmarks\train\image-tensors-selfsupervision',
                                    image_size=84):
    resize = transforms.Compose(
        [
            transforms.Resize(image_size * 3),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
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
    index = pd.read_csv(source)
    for i, row in enumerate(index.iterrows()):
        done = False
        url = None
        attempts = 0
        while not done:
            attempts += 1
            try:
                label = str(row[1]['id'])
                label_folder = os.path.join(target, label)

                url = str(row[1]['url'])

                os.makedirs(label_folder, exist_ok=True)

                response = requests.get(url)
                pil_image = Image.open(BytesIO(response.content))
                pil_image = pil_image.convert('RGB')
                for crop_i in range(CROPS_NUM):
                    image_id = str(crop_i)
                    image_path = os.path.join(label_folder, image_id + r'.pt')
                    if os.path.exists(image_path):
                        break
                    image_tensor = transform(pil_image)
                    torch.save(image_tensor, image_path)

                done = True
            except (OSError, AttributeError):
                if attempts <= ATTEMPTS:
                    print("Error with url %s, delay for %d second(s)" % (url, SLEEP))
                    time.sleep(SLEEP)
                else:
                    done = True
                    print("Skipped")

        if i % 50 == 0:
            print(i)


def remove_small(threshold, root=r"C:\datasets\google-landmarks\train\image-tensors-selfsupervision"):
    for label in os.listdir(root):
        path = os.path.join(root, label)
        cnt = len(os.listdir(path))
        if cnt < threshold:
            print(label, cnt)
            shutil.rmtree(path)
            print("Deleted!")

# if __name__ == '__main__':
#     # load_from_index(image_size=84)
#     load_from_selfsupervision_index(image_size=84)
#     remove_small(8)
