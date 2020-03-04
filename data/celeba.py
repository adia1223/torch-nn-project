from data import ImageDirDataset
from data.image_transforms import TO_GRAYSCALE_TENSOR


class CelebaCroppedDataset(ImageDirDataset):
    def __init__(self, data_dir="C:\\datasets\\celeba\\images\\img_align_celeba", transform=TO_GRAYSCALE_TENSOR,
                 preload_to_ram=False):
        super().__init__(data_dir, transform, preload_to_ram=preload_to_ram)
