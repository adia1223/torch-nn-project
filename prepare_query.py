import argparse

import torch
from PIL import Image
from torchvision import transforms

from data.image_transforms import TO_RGB_TENSOR


def parse_args():
    parser = argparse.ArgumentParser(description='Convert image to tensor')

    parser.add_argument('file', type=argparse.FileType('rb'), help='Path to file')
    parser.add_argument('--output_file', type=argparse.FileType('wb'), help='Path to result file', default='query.pt')
    parser.add_argument('--image_resize', type=int, help='Size of scaled query (default = 84)', default=84)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    source = args.file
    resize_transform = transforms.Compose(
        [transforms.Resize(args.image_resize), transforms.CenterCrop(args.image_resize)]
    )

    # with open(source, 'rb') as f:
    img = Image.open(source)
    img = img.convert('RGB')
    img = resize_transform(img)
    tensor = TO_RGB_TENSOR(img)

    torch.save(tensor, args.output_file)
