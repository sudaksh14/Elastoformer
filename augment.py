# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
3Augment implementation
Data-augmentation (DA) based on dino DA (https://github.com/facebookresearch/dino)
and timm DA (https://github.com/rwightman/pytorch-image-models)
"""
import torch
from torchvision import transforms
import numpy as np
import random

from PIL import ImageFilter, ImageOps

# ---- Custom Augs ----
class GaussianBlur(object):
    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if random.random() > self.prob:
            return img
        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        return img


class GrayScale(object):
    def __init__(self, p=0.2):
        self.p = p
        self.transf = transforms.Grayscale(3)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        return img


class HorizontalFlip(object):
    def __init__(self, p=0.2):
        self.p = p
        self.transf = transforms.RandomHorizontalFlip(p=1.0)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        return img


# ---- Main Augmentation Pipeline ----
def new_data_aug_generator(args=None):
    img_size = args.input_size
    remove_random_resized_crop = True
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    # Primary transforms
    if remove_random_resized_crop:
        primary_tfl = [
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(img_size, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip()
        ]
    else:
        primary_tfl = [
            transforms.RandomResizedCrop(
                img_size, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip()
        ]

    # Secondary transforms (color/blur/solarize/gray)
    secondary_tfl = [
        transforms.RandomChoice([
            GrayScale(p=1.0),
            Solarization(p=1.0),
            GaussianBlur(p=1.0)
        ])
    ]

    if args.color_jitter is not None and args.color_jitter != 0:
        secondary_tfl.append(
            transforms.ColorJitter(args.color_jitter, args.color_jitter, args.color_jitter)
        )

    # Final transforms (tensor + normalization)
    final_tfl = [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]

    return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)