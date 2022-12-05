import glob
import os
import random

import PIL
import albumentations as A
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL.Image import Image
from torch.utils.data import Dataset


class TwoClassDataset(Dataset):

    def check_dataset_folder(self):
        if not os.path.isdir(self.class_one_path):
            raise PathException(f"The given path is not a valid: {self.class_one_path}")
        if not os.path.isdir(self.class_two_path):
            raise PathException(f"The given path is not a valid: {self.class_two_path}")

    def log(self, message):
        if self.logs:
            print(message)

    def __init__(self, path_class_one: str, path_class_two: str, resize=256, max_num_images=-1, logs=True):
        self.class_one_path = path_class_one
        self.class_two_path = path_class_two
        self.resize_size = resize
        self.logs = logs
        self.log('init custom image-sketch dataset')

        self.check_dataset_folder()

        # get images
        self.log('scanning class one')
        self.paths_images_class_one = []
        self.paths_images_class_one.extend(glob.glob(os.path.join(path_class_one, '*.png')))
        self.paths_images_class_one.extend(glob.glob(os.path.join(path_class_one, '*.jpg')))
        self.paths_images_class_one.extend(glob.glob(os.path.join(path_class_one, '*.jpeg')))
        self.log(f'found {len(self.paths_images_class_one)} images')

        # get sketches
        self.log('scanning the sketches')
        self.path_images_class_two = []
        self.path_images_class_two.extend(glob.glob(os.path.join(path_class_two, '*.png')))
        self.path_images_class_two.extend(glob.glob(os.path.join(path_class_two, '*.jpg')))
        self.path_images_class_two.extend(glob.glob(os.path.join(path_class_two, '*.jpeg')))
        self.log(f'found {len(self.path_images_class_two)} images')

        self.apply_transform = False

        if max_num_images > 0:
            self.log(f'limiting number of images to {max_num_images} per class')
            self.paths_images_class_one = self.paths_images_class_one[:max_num_images]
            self.path_images_class_two = self.path_images_class_two[:max_num_images]
        self.log('Finished initializing the Dataset')

        # check whether we find a sketch for each image
        # assert len(self.paths_images_class_one) == len(self.path_images_class_two)

        self.size = len(self.paths_images_class_one) + len(self.path_images_class_two)
        self.log(f"A total of {self.size} images were found for this dataset")

        # combine both
        self.path_images = self.paths_images_class_one + self.path_images_class_two
        self.labels = [0 for _ in range(len(self.paths_images_class_one))] \
                      + [1 for _ in range(len(self.path_images_class_two))]

    def __len__(self):
        return self.size

    def preprocess(self, img):
        s = min(img.size)

        if s < self.resize_size:
            raise ValueError(
                f'Image is smaller than the resized size. Expected at least {self.resize_size} width (or height) but got {s}')

        r = self.resize_size / s
        s = (round(r * img.size[1]), round(r * img.size[0]))
        img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
        img = TF.center_crop(img, output_size=2 * [self.resize_size])
        img = torch.unsqueeze(T.ToTensor()(img), 0)
        return img

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.path_images[idx]
        image = PIL.Image.open(img_path).convert('RGB')
        image = self.preprocess(image)

        label = self.labels[idx]

        return image, torch.tensor([label])


class PathException(Exception):
    def __init__(self, string):
        super(PathException, self).__init__(string)
