# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from torchvision import datasets, transforms
import torch
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform, Mixup
from fastdownload import FastDownload
from augment import new_data_aug_generator
from torch.utils.data import DataLoader, Subset
from sampler import RASampler


train_path = "/home/dkoelma1/VisualSearch/Imagenet_train.tar"
val_path = "/home/dkoelma1/VisualSearch/Imagenet_val.tar"

imagenet_path = "/ssdstore/ImageNet/"

# ----- Mixup + CutMix -----
mixup_fn = Mixup(
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    cutmix_minmax=None,
    prob=1.0,
    switch_prob=0.5,  # probability to switch between mixup and cutmix
    mode='batch',
    label_smoothing=0.11,
    num_classes=1000
)

def build_dataset(is_train, args):
    if is_train:
        transform = new_data_aug_generator(args)
    else:
        transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(
            args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
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
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            # to maintain same ratio w.r.t. 224 images
            transforms.Resize(size, interpolation=3),
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def load_imagenet(
        data_set='IMNET',
        datapath="/var/scratch/dchabal/quokka/data/imagenet",
        input_size=224,
        color_jitter=.3,
        aa='rand-m9-mstd0.5-inc1',
        train_interpolation='bicubic',
        reprob=.25,
        remode='pixel',
        recount=1,
        eval_crop_ratio=0.875,
        batch_size=128,
        num_workers=16, distributed=True, ra_sampler=True, ra_reps=3, debug=False):
    class Args:
        pass
    args = Args()
    args.data_set = data_set
    args.data_path = datapath
    args.input_size = input_size
    args.color_jitter = color_jitter
    args.aa = aa
    args.train_interpolation = train_interpolation
    args.reprob = reprob
    args.remode = remode
    args.recount = recount
    args.eval_crop_ratio = eval_crop_ratio
    args.distributed = distributed
    args.ra_sampler = ra_sampler
    args.ra_reps = ra_reps
    
    
    train_dataset, num_classes = build_dataset(is_train=True, args=args)
    val_dataset, _ = build_dataset(is_train=False, args=args)
    
    if debug:
        print("Debug mode: using smaller datasets")
        train_dataset = Subset(train_dataset, indices=torch.randperm(len(train_dataset))[:4000])
        val_dataset = Subset(val_dataset, indices=torch.randperm(len(val_dataset))[:1000])
    
    if args.distributed:    
        if args.ra_sampler:
            train_sampler = RASampler(train_dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)


    train_dataset = DataLoader(
        train_dataset, sampler=train_sampler,
        batch_size=batch_size,
        num_workers=num_workers, pin_memory=True, drop_last=True)

    val_dataset = DataLoader(
        val_dataset, sampler=val_sampler,
        batch_size=batch_size,
        num_workers=num_workers, pin_memory=True, drop_last=False)


    return train_dataset, val_dataset, num_classes


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

def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_train from {cache_path}")
        # TODO: this could probably be weights_only=True
        dataset, _ = torch.load(cache_path, weights_only=False)
    else:
        # We need a default value for the variables below because args may come
        # from train_quantization.py which doesn't define them.
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        ra_magnitude = getattr(args, "ra_magnitude", None)
        augmix_severity = getattr(args, "augmix_severity", None)
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            imgaug_presets.ClassificationPresetTrain(
                crop_size=train_crop_size,
                interpolation=interpolation,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,
                ra_magnitude=ra_magnitude,
                augmix_severity=augmix_severity,
                backend=args.backend,
                use_v2=args.use_v2,
            ),
        )
        if args.cache_dataset:
            print(f"Saving dataset_train to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_test from {cache_path}")
        # TODO: this could probably be weights_only=True
        dataset_test, _ = torch.load(cache_path, weights_only=False)
    else:
        if args.weights and args.test_only:
            weights = torchvision.models.get_weight(args.weights)
            preprocessing = weights.transforms(antialias=True)
            if args.backend == "tensor":
                preprocessing = torchvision.transforms.Compose([torchvision.transforms.PILToTensor(), preprocessing])

        else:
            preprocessing = imgaug_presets.ClassificationPresetEval(
                crop_size=val_crop_size,
                resize_size=val_resize_size,
                interpolation=interpolation,
                backend=args.backend,
                use_v2=args.use_v2,
            )

        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            preprocessing,
        )
        if args.cache_dataset:
            print(f"Saving dataset_test to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


if __name__ == '__main__':
    download_datasets("imagenette")