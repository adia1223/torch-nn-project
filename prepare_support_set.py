import argparse
import os

import torch
from PIL import Image
from torchvision import transforms

from data.image_transforms import TO_RGB_TENSOR


def parse_dir(path):
    if not os.path.exists(path):
        msg = "'%s' not found" % path
        raise argparse.ArgumentTypeError(msg)

    if not os.path.isdir(path):
        msg = "'%s' is not directory" % path
        raise argparse.ArgumentTypeError(msg)

    data = []

    for class_dir in [e.path for e in os.scandir(path) if e.is_dir()]:
        data.append([])
        for f in [e.path for e in os.scandir(class_dir) if e.is_file()]:
            with open(f, 'rb') as inp:
                image = Image.open(inp)
                image = image.convert('RGB')
            data[-1].append(image)
        if len(data[0]) != len(data[-1]):
            msg = "'%s' classes are not equal" % path
            raise argparse.ArgumentTypeError(msg)

    if len(data) < 2:
        msg = "'%s' has less than 2 classes" % path
        raise argparse.ArgumentTypeError(msg)

    return data


def parse_args():
    parser = argparse.ArgumentParser(description='Compose support dataset to tensor.')

    parser.add_argument('folder', type=parse_dir, help='Path to dataset_folder')
    parser.add_argument('--output_file', type=argparse.FileType('wb'), help='Path to result file', default='query.pt')
    parser.add_argument('--image_resize', type=int, help='Size of scaled query (default = 84)', default=84)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    data = args.folder
    resize_transform = transforms.Compose(
        [transforms.Resize(args.image_resize), transforms.CenterCrop(args.image_resize)]
    )

    for i in range(len(data)):
        for j in range(len(data[i])):
            img = data[i][j]
            img = resize_transform(img)
            data[i][j] = TO_RGB_TENSOR(img)
        data[i] = torch.stack(data[i])
    tensor = torch.stack(data)
    print("%d classes, %d shots" % (tensor.size(0), tensor.size(1)))
    torch.save(tensor, args.output_file)
