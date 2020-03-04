from torchvision import transforms

RGB_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

TO_RGB_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*RGB_NORMALIZATION)
])

TO_GRAYSCALE_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])


def scaling(size):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.CenterCrop(size),
        TO_RGB_TENSOR
    ])


def scaling_pil(size):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        TO_RGB_TENSOR
    ])


def scaling_grayscale(size):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.Grayscale(),
        TO_GRAYSCALE_TENSOR
    ])
