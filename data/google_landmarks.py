import os
import random
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


class GoogleLandmarksDataset(data.LabeledDataset):

    def __init__(self, root=r"C:\datasets\google-landmarks\train\image-tensors", reduce=0.0,
                 random_seed=42, **kwargs):
        self.CLASSES = len(os.listdir(root))

        self.reduce = reduce
        random.seed(random_seed)

        self.source_dataset_train = torchvision.datasets.DatasetFolder(root=root, loader=tensor_loader,
                                                                       extensions=('pt',))

        self.dataset_train_size = len(self.source_dataset_train)
        items = []
        labels = []
        for i in range(self.dataset_train_size):
            items.append(ImageItem(self.source_dataset_train, i))
            labels.append(self.source_dataset_train[i][1])
        is_test = [0] * self.dataset_train_size

        super(GoogleLandmarksDataset, self).__init__(items, labels, is_test)

        self.train_subdataset, self.test_subdataset = self.subdataset.train_test_split()

        if reduce < 1:
            self.train_subdataset = self.train_subdataset.downscale(1 - reduce)
        else:
            self.train_subdataset = self.train_subdataset.balance(reduce)

    def __getitem__(self, item):
        image, label, is_test = super(GoogleLandmarksDataset, self).__getitem__(item)
        return image, label, is_test

    def label_stat(self):
        pass

    def train(self):
        return self.train_subdataset

    def test(self):
        return self.test_subdataset


import pandas as pd

ATTEMPTS = 3
SLEEP = 1


def load_from_index(source=r'C:\datasets\google-landmarks\train\filtered_train.csv',
                    target=r'C:\datasets\google-landmarks\train\image-tensors',
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
            except OSError:
                if attempts <= ATTEMPTS:
                    print("Error with url %s, delay for %d second(s)" % (url, SLEEP))
                    time.sleep(SLEEP)
                else:
                    done = True
                    print("Skipped")

        if i % 50 == 0:
            print(i)


if __name__ == '__main__':
    load_from_index()
