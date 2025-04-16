# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform
from fastdownload import FastDownload


train_path = "/home/dkoelma1/VisualSearch/Imagenet_train.tar"
val_path = "/home/dkoelma1/VisualSearch/Imagenet_val.tar"

imagenet_path = "/ssdstore/ImageNet/"

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:  
            t.append(
            transforms.Resize((args.input_size, args.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def download_datasets(dataset_name, root_dir="../Datasets"):
    os.makedirs(root_dir, exist_ok=True)
    
    transform = transforms.ToTensor()

    if dataset_name.lower() == "cifar10":
        datasets.CIFAR10(root=root_dir, train=True, download=True, transform=transform)
        datasets.CIFAR10(root=root_dir, train=False, download=True, transform=transform)
        print("CIFAR-10 downloaded.")
    
    elif dataset_name.lower() == "cifar100":
        datasets.CIFAR100(root=root_dir, train=True, download=True, transform=transform)
        datasets.CIFAR100(root=root_dir, train=False, download=True, transform=transform)
        print("CIFAR-100 downloaded.")

    elif dataset_name.lower() == "tiny_imagenet":
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        target_path = os.path.join(root_dir, "tiny-imagenet-200.zip")
        if not os.path.exists(target_path):
            import urllib.request, zipfile
            print("Downloading Tiny ImageNet...")
            urllib.request.urlretrieve(url, target_path)
            with zipfile.ZipFile(target_path, 'r') as zip_ref:
                zip_ref.extractall(root_dir)
            print("Tiny ImageNet downloaded and extracted.")
        else:
            print("Tiny ImageNet already exists.")

    elif dataset_name.lower() == "imagenette":
        fd = FastDownload(base=root_dir)
        url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz'  # 160px version
        path = fd.get(url)
        print(f"Imagenette downloaded at: {path}")

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == '__main__':
    download_datasets("imagenette")