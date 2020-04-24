import os
import random

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

import data

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

resize = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
    ]
)

augment = transforms.Compose(
    [
        transforms.RandomRotation(degrees=15),
        transforms.RandomPerspective(p=0.75),
    ]
)

normalize = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
)

CLASSES = 43


class ImageItem(object):
    def __init__(self, file):
        self.file = file

    def load(self):
        img = Image.open(self.file)
        return img


class GTSRBDataset(data.LabeledDataset):
    CLASSES = 43

    def __init__(self, data_dir="C:\\datasets\\gtsrb-german-traffic-sign", augment_prob=0.0, reduce=0.0,
                 random_seed=42, **kwargs):
        self.reduce = reduce
        random.seed(random_seed)

        self.dir = data_dir
        self.test_transform = transforms.Compose(
            [
                resize,
                normalize
            ]
        )
        self.train_transform = transforms.Compose(
            [
                resize,
                transforms.RandomApply([augment], p=augment_prob),
                normalize
            ]
        )

        self.train_data_file = os.path.join(data_dir, "Train.csv")
        self.test_data_file = os.path.join(data_dir, "Test.csv")

        train_data = pd.read_csv(self.train_data_file)
        files = [os.path.join(data_dir, x) for x in train_data['Path']]
        labels = list(train_data['ClassId'])
        is_test = [0] * len(train_data)

        test_data = pd.read_csv(self.test_data_file)
        files += [os.path.join(data_dir, x) for x in test_data['Path']]
        labels += list(test_data['ClassId'])
        is_test += [1] * len(test_data)

        super(GTSRBDataset, self).__init__(list(map(ImageItem, files)), labels, is_test)

        self.train_subdataset, self.test_subdataset = self.subdataset.train_test_split()

        if reduce < 1:
            self.train_subdataset = self.train_subdataset.downscale(1 - reduce)
        else:
            self.train_subdataset = self.train_subdataset.balance(reduce)

    def __getitem__(self, item):
        image, label, is_test = super(GTSRBDataset, self).__getitem__(item)
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


def save_as_tensors(source='C:\\datasets\\mini-imagenet\\train', target=r'C:\datasets\mini-imagenet\train_tensors',
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
            target_file = os.path.join(cur_target, image.replace('.JPEG', '.pt'))
            with open(source_image_file, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
                image_tensor = transform(img)
                torch.save(image_tensor, target_file)
        print(class_label, i)

# if __name__ == '__main__':
#     save_as_tensors()
