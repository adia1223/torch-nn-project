import random
from os import listdir
from os.path import isfile, join

import imageio
import torch
from torch.utils.data import Dataset


class ImageDirDataset(Dataset):
    def __init__(self, data_dir, transform, preload_to_ram=False):
        self.dir = data_dir
        self.transform = transform
        self.files = self.get_files()
        self.data = None
        if preload_to_ram:
            self.data = [self.image_to_tensor(file) for file in self.files]

    def __getitem__(self, index):
        if self.data is None:
            sample = self.image_to_tensor(self.files[index])
        else:
            sample = self.data[index]

        return sample

    def __len__(self):
        return len(self.files)

    def get_files(self):
        files = [join(self.dir, f) for f in listdir(self.dir) if isfile(join(self.dir, f))]
        return files

    def image_to_tensor(self, path):
        img = imageio.imread(path)
        return self.transform(img)


class LabeledSubdataset(Dataset):
    def __init__(self, base, indices):
        self.base_dataset = base
        self.indices = indices
        self.set_indices = set(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.base_dataset[self.indices[index]]

    def labels(self):
        result = set()
        for i in self.indices:
            label = self.base_dataset.get_label(i)
            result.add(label)

        return result

    def balance(self, n_items):
        random.shuffle(self.indices)

        balanced_classes = {}
        for i in self.indices:
            label = self.base_dataset.get_label(i)
            if label not in balanced_classes:
                balanced_classes[label] = [i]
            elif len(balanced_classes[label]) < n_items:
                balanced_classes[label].append(i)
        new_indices = []

        for class_label in balanced_classes.keys():
            new_indices += balanced_classes[class_label]

        return LabeledSubdataset(self.base_dataset, new_indices)

    def extract_balanced(self, n_items):
        random.shuffle(self.indices)

        balanced_classes = {}
        not_extracted_indices = []
        for i in self.indices:
            label = self.base_dataset.get_label(i)
            if label not in balanced_classes:
                balanced_classes[label] = [i]
            elif len(balanced_classes[label]) < n_items:
                balanced_classes[label].append(i)
            else:
                not_extracted_indices.append(i)
        new_indices = []

        for class_label in balanced_classes.keys():
            new_indices += balanced_classes[class_label]
        # print(len(self.indices), len(new_indices), len(not_extracted_indices))

        return LabeledSubdataset(self.base_dataset, new_indices), LabeledSubdataset(self.base_dataset,
                                                                                    not_extracted_indices)

    def downscale(self, k):
        classes = {}
        for i in self.indices:
            label = self.base_dataset.get_label(i)
            if label not in classes:
                classes[label] = [i]
            else:
                classes[label].append(i)

        new_indices = []

        for class_label in classes.keys():
            size = len(classes[class_label])
            size = int(size * k)
            new_indices += random.sample(classes[class_label], size)

        return LabeledSubdataset(self.base_dataset, new_indices)

    def extract_classes(self, classes_cnt):
        class_labels = list(set(self.base_dataset.get_label(i) for i in self.indices))
        extracted_classes = set(random.sample(class_labels, classes_cnt))

        extracted_indices = []
        other_indices = []

        for i in self.indices:
            if self.base_dataset.get_label(i) in extracted_classes:
                extracted_indices.append(i)
            else:
                other_indices.append(i)

        return LabeledSubdataset(self.base_dataset, extracted_indices), LabeledSubdataset(self.base_dataset,
                                                                                          other_indices)

    def __contains__(self, item):
        return item in self.set_indices

    def extract_samples(self, samples_per_class):
        extracted = self.balance(samples_per_class)

        other_indices = []
        for i in self.indices:
            if i not in extracted:
                other_indices.append(i)

        return extracted, LabeledSubdataset(self.base_dataset,
                                            other_indices)

    def train_test_split(self):
        train = []
        test = []
        for i in self.indices:
            is_test = self.base_dataset.get_is_test(i)
            if is_test:
                test.append(i)
            else:
                train.append(i)
        return LabeledSubdataset(self.base_dataset, train), LabeledSubdataset(self.base_dataset, test)

    def set_test(self, value):
        for index in self.indices:
            self.base_dataset.data[index][2] = value

    def random_batch(self, size):
        batch_indices = random.sample(self.indices, size)
        items = []
        labels = []
        for i in batch_indices:
            item, label, _ = self.base_dataset[i]
            items.append(item)
            labels.append(label)
        return torch.stack(items), torch.tensor(labels)

    def balanced_batch(self, per_class):
        classes = {}
        for i in self.indices:
            label = self.base_dataset.get_label(i)
            if label not in classes:
                classes[label] = [i]
            else:
                classes[label].append(i)

        indices = []

        for label in classes:
            # print('\t', len(classes[label]))
            class_indices = random.sample(classes[label], per_class)
            indices += class_indices

        items = []
        labels = []
        for i in indices:
            item, label, _ = self.base_dataset[i]
            items.append(item)
            labels.append(label)
        return torch.stack(items), torch.tensor(labels)


class LabeledDataset(Dataset):
    def __init__(self, items, labels, test):
        self.classes = len(set(labels))
        self.data = list(map(list, zip(items, labels, test)))
        self.subdataset = LabeledSubdataset(self, list(range(len(self.data))))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0].load(), self.data[index][1], self.data[index][2]

    def get_label(self, index):
        return self.data[index][1]

    def get_object(self, index):
        return self.data[index][0]

    def get_is_test(self, index):
        return self.data[index][2]


from data.cifar10 import CIFAR10Dataset, CIFAR100Dataset
from data.gtsrb import GTSRBDataset
from data.cub import CUBDataset
from data.mini_imagenet import MiniImageNetDataset
from data.google_landmarks import GoogleLandmarksDataset, GoogleLandmarksDataset2, \
    GoogleLandmarksDatasetSelfSupervision, GoogleLandmarksDatasetTest
from data.taco import TacoDataset

LABELED_DATASETS = {
    'cifar10': CIFAR10Dataset,
    'cifar100': CIFAR100Dataset,
    'gtsrb': GTSRBDataset,
    'cub': CUBDataset,
    'miniImageNet': MiniImageNetDataset,
    'taco': TacoDataset,
    'google-landmarks': GoogleLandmarksDataset,
    'google-landmarks-2': GoogleLandmarksDataset2,
    'google-landmarks-selfsupervision': GoogleLandmarksDatasetSelfSupervision,
    'google-landmarks-test': GoogleLandmarksDatasetTest,
}
