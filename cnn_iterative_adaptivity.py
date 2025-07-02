import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
from typing import Dict, List, Optional, Set, Tuple, Union
import torch
import torchvision
import torch.nn.functional as F
import torch_pruning as tp
from models.Resnet import *
from models.VGG import *
import warnings
from torchvision.datasets import ImageFolder
import torchvision.datasets as datasets
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import torchvision.models as models
from torch.utils.data import SubsetRandomSampler, Subset
from tqdm import tqdm
import imgaug_presets
from copy import deepcopy
# from models.hf_vit import create_vit_general
from prune_utils import *
from cnn_prune_utils import get_resnet_layer_sizes, extract_weight_subsets, extract_core_weights, get_model_info_core, get_model_info_resnet, update_global_weights, freeze_partial_weights_cnn, inject_stochastic_depth_resnet, get_model_info_vgg
from trainer import fine_tuner, evaluate, fine_tuner_core
from sampler import RASampler
import utils
import math
import argparse
from aug_transforms import get_mixup_cutmix
from torch.utils.data.dataloader import default_collate

torch.manual_seed(42)

def get_args_parser(add_help=True):

    parser = argparse.ArgumentParser(description="ViT Pruning and PyTorch Classification Training", add_help=add_help)

    ###################################################################################################################
    #                                                   PRUNING                                                       #
    ###################################################################################################################

    parser.add_argument('--exp_name', default='ViT_Adaptivity', type=str, help='Name of the experiment')
    parser.add_argument('--dataset_name', default='imagenet', type=str, help='Dataset used')
    parser.add_argument('--model_name', default='google/vit-base-patch16-224', type=str, help='model name')
    parser.add_argument('--data_path', default='data/imagenet', type=str, help='model name')
    parser.add_argument('--taylor_batchs', default=10, type=int, help='number of batchs for taylor criterion')
    parser.add_argument('--pruning_ratio', default=0.5, type=float, help='prune ratio')
    parser.add_argument('--iterative', default=False, action='store_true', help='True for Iterative Pruning')
    parser.add_argument('--pruning_steps', default=1, type=int, help='number of Prune steps/Adaptive modes')
    parser.add_argument('--bottleneck', default=False, action='store_true', help='bottleneck or uniform')
    parser.add_argument('--pruning_type', default='l1', type=str, help='pruning type', choices=['random', 'taylor', 'l1', 'l2', 'hessian'])
    parser.add_argument('--test_accuracy', default=False, action='store_true', help='test accuracy')
    parser.add_argument('--global_pruning', default=False, action='store_true', help='enables global pruning(global_compression = 1-pruning_ratio)')
    parser.add_argument('--isomorphic', default=False, action='store_true', help='enables isomorphic pruning ECCV 2024, https://arxiv.org/abs/2407.04616, overides global_pruning')
    parser.add_argument('--rebuild', default=False, action='store_true', help='Rebuilding for adaptivity')
    
    parser.add_argument('--train_batch_size', default=128, type=int, help='train batch size')
    parser.add_argument('--val_batch_size', default=128, type=int, help='val batch size')
    parser.add_argument('--save_as', default=None, type=str, help='save as')
    parser.add_argument('--debug', default=False, action='store_true', help='Use for dubugging')

    ###################################################################################################################
    #                                                 FINE-TUNING                                                     #
    ###################################################################################################################
    parser.add_argument("--data-path", default="/ssdstore/ImageNet/", type=str, help="dataset path")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=512, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=20, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--core_epochs", default=50, type=int, metavar="N", help="number of total epochs for the core model training")
    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="adamw", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.003, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--dropout", default=0.0, type=float, help="dropout rate")
    parser.add_argument(
        "--core_weight_decay",
        default=0.3,
        type=float,
        metavar="W",
        help="Core Training weight decay (default: 0.3)",
        dest="core_weight_decay",
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=0,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.11, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--mixup-alpha", default=0.2, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=1.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="cosineannealinglr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=5, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="linear", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.033, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="./saves/Checkpoints/", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # Data augmentation parameters
    parser.add_argument("--auto-augment", default="ra", type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.25, type=float, help="random erasing probability (default: 0.0)")
    parser.add_argument("--stochastic_depth", action="store_true", help="Use Stochastic Depth for training, as implemented in https://arxiv.org/abs/1603.09382")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--distributed", action="store_true", help="Use distributed training")
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")
    parser.add_argument("--log_wandb", action="store_true", help="Use Weights and Bias to log the training curves")
    return parser


def prepare_imagenet(imagenet_root, train_batch_size=64, val_batch_size=128, num_workers=4, use_imagenet_mean_std=True, debug=False):
    """The imagenet_root should contain train and val folders.
    """

    print('Parsing dataset...')

    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)
    ra_magnitude = getattr(args, "ra_magnitude", None)
    augmix_severity = getattr(args, "augmix_severity", None)

    train_dst = ImageFolder(os.path.join(imagenet_root, 'train'),
                            transform=imgaug_presets.ClassificationPresetTrain(
                            crop_size=224,
                            interpolation=InterpolationMode.BILINEAR,
                            auto_augment_policy=auto_augment_policy,
                            random_erase_prob=random_erase_prob,
                            ra_magnitude=ra_magnitude,
                            augmix_severity=augmix_severity,
                            backend=args.backend,
                            use_v2=args.use_v2,
                        ),
    )
    

    val_dst = ImageFolder(os.path.join(imagenet_root, 'val'), 
                          transform=imgaug_presets.ClassificationPresetEval(
                                mean=[0.485, 0.456, 0.406] if use_imagenet_mean_std else [0.5, 0.5, 0.5],
                                std=[0.229, 0.224, 0.225] if use_imagenet_mean_std else [0.5, 0.5, 0.5],
                                crop_size=224,
                                resize_size=256,
                                interpolation=InterpolationMode.BILINEAR,
                            )
    )

    if debug:
        train_dst = Subset(train_dst, indices=torch.randperm(len(train_dst))[:500])
        val_dst = Subset(val_dst, indices=torch.randperm(len(val_dst))[:100])

    # ADD CUTMIX AND MIXUP AUGMENTATIONS
    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, num_classes=1000, use_v2=args.use_v2
    )
    if mixup_cutmix is not None:

        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))

    else:
        collate_fn = default_collate

    if args.distributed:    
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(train_dst, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dst)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dst, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dst)
        val_sampler = torch.utils.data.SequentialSampler(val_dst)


    train_loader = torch.utils.data.DataLoader(train_dst, batch_size=train_batch_size, sampler=train_sampler, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dst, batch_size=val_batch_size, sampler=val_sampler, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, train_sampler, val_sampler

def prepare_imagenette(path="../Datasets/data/imagenette2-160"):
    
    print('Parsing Imagenette dataset...')

    train_transforms = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

    val_transforms = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(path, 'train')
    val_dir = os.path.join(path, 'val')

    train_dataset = ImageFolder(root=train_dir, transform=train_transforms)
    val_dataset = ImageFolder(root=val_dir, transform=val_transforms)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    print(train_dataset.classes)

    if args.distributed:
        return train_loader, val_loader, train_sampler, val_sampler
    else:
        return train_loader, val_loader, None, None
    
    
def get_cifar_dataloaders(dataset='cifar10', data_root='../Datasets/data', batch_size=128, num_workers=4, distributed=True):
    """
    Returns train and test loaders for CIFAR-10 or CIFAR-100 with ImageNet-compatible transforms.
    """

    # ImageNet mean and std
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Transform for training and testing
    transform_train = T.Compose([
        T.Resize(224),
        T.RandomHorizontalFlip(),         
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    transform_test = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])


    if dataset.lower() == 'cifar10':
        train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
        num_classes = 10
    elif dataset.lower() == 'cifar100':
        train_dataset = datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform_test)
        num_classes = 100
    else:
        raise ValueError("Dataset must be either 'cifar10' or 'cifar100'.")

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None

    # ADD CUTMIX AND MIXUP AUGMENTATIONS
    mixup_cutmix = get_mixup_cutmix(mixup_alpha=0.8, cutmix_alpha=1.0, num_classes=num_classes, use_v2=False)
    if mixup_cutmix is not None:
        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))
    else:
        collate_fn = default_collate

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             sampler=test_sampler, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader, train_sampler, test_sampler, num_classes


######################################################################## MAIN STARTS ###########################################################################

def main(args):

    if args.distributed:
            utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    example_inputs = torch.randn(1,3,224,224).to(device)

    if args.pruning_type == 'random':
        imp = tp.importance.RandomImportance()
    elif args.pruning_type == 'taylor':
        imp = tp.importance.GroupTaylorImportance()
    elif args.pruning_type == 'l2':
        imp = tp.importance.GroupMagnitudeImportance(p=2)
    elif args.pruning_type == 'l1':
        imp = tp.importance.GroupMagnitudeImportance(p=1)
    elif args.pruning_type == 'hessian':
        imp = tp.importance.GroupHessianImportance()
    else: raise NotImplementedError

    if args.dataset_name.startswith('imagenet'):
        train_loader, val_loader, train_sampler, val_sampler = prepare_imagenet(args.data_path, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size, debug=args.debug)
        num_classes = 1000
        # train_loader, val_loader, train_sampler, val_sampler = prepare_imagenette()
    if args.dataset_name.startswith('cifar'):
        train_loader, val_loader, train_sampler, val_sampler, num_classes = get_cifar_dataloaders(dataset=args.dataset_name, batch_size=args.train_batch_size, distributed=args.distributed)

    # Load the model
    model = torchvision.models.get_model(args.model_name, weights=args.weights, num_classes=1000)
    if args.dataset_name.startswith('cifar'):
        if args.model_name.startswith('resnet'):
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        else:
            model.classifier[-1] = torch.nn.Linear(4096, num_classes)
    model = model.to(device)


    for p in model.parameters():
        p.requires_grad_(True)

    if False:
        print_trainable_summary(model)
        # for name, param in model.named_parameters():
        #     print(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'} - {param.numel():,} params")

        if args.test_accuracy:
            criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
            print("Testing accuracy of the original model...")
            acc_ori, loss_ori = evaluate(model, criterion, val_loader, device=device, dist=args.distributed, CNN=True)
            print("Accuracy: %.4f, Loss: %.4f"%(acc_ori, loss_ori))

        fine_tuner(args, device, model, train_loader, val_loader, train_sampler, val_sampler)
        print("Fine-tuning complete")
        if args.test_accuracy:
            criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
            print("Testing accuracy of the original model...")
            acc_ori, loss_ori = evaluate(model, criterion, val_loader, device=device, dist=args.distributed, CNN=True)
            print("Accuracy: %.4f, Loss: %.4f"%(acc_ori, loss_ori))
    
    orig_copy = deepcopy(model)

    orig_statedict = orig_copy.state_dict()
    orig_dimensions = get_resnet_layer_sizes(orig_statedict)

    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
    base_size = get_model_size_mb_multi(model)
    print(f"Orig MAC's: {base_macs/1e9:.2f} G, Orig Params: {base_params/1e6:.2f} M, Orig Size: {base_size:.2f} MB")
    
    if args.test_accuracy:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        print("Testing accuracy of the original model...")
        acc_ori, loss_ori = evaluate(orig_copy, criterion, val_loader, device=device, dist=args.distributed, CNN=True)
        print("Accuracy: %.4f, Loss: %.4f"%(acc_ori, loss_ori))
    
    print("Pruning %s..."%args.model_name)
    ignored_layers = []
    for m in model.modules():
        # if isinstance(m, nn.Linear) and m.out_features == num_classes:
        #     ignored_layers.append(m)
        if isinstance(m, nn.Linear):
            ignored_layers.append(m)

    print("Ignored Layers:", ignored_layers)
                
    pruner = tp.pruner.BasePruner(
                model, 
                example_inputs,
                isomorphic=args.isomorphic,
                global_pruning=args.global_pruning, # If False, a uniform pruning ratio will be assigned to different layers.
                importance=imp, # importance criterion for parameter selection
                iterative_steps=args.pruning_steps, # number of pruning steps
                pruning_ratio=args.pruning_ratio, # target pruning ratio
                ignored_layers=ignored_layers,
    )

    if isinstance(imp, (tp.importance.GroupTaylorImportance, tp.importance.GroupHessianImportance)):
            model.zero_grad()
            if isinstance(imp, tp.importance.GroupHessianImportance):
                imp.zero_grad()
            print("Accumulating gradients for pruning...")
            for k, (imgs, lbls) in enumerate(train_loader):
                if k>=args.taylor_batchs: break
                imgs = imgs.to(device)
                lbls = lbls.to(device)
                output = model(imgs).logits
                if isinstance(imp, tp.importance.GroupHessianImportance):
                    loss = torch.nn.functional.cross_entropy(output, lbls, reduction='none')
                    for l in loss:
                        model.zero_grad()
                        l.backward(retain_graph=True)
                        imp.accumulate_grad(model)
                elif isinstance(imp, tp.importance.GroupTaylorImportance):
                    loss = torch.nn.functional.cross_entropy(output, lbls)
                    loss.backward()

    # Pruning Metadata for Rebuilding
    pruned_index_out = [{} for _ in range(args.pruning_steps)]
    pruned_index_in = [{} for _ in range(args.pruning_steps)]

    non_pruned_index_out = [{} for _ in range(args.pruning_steps)]
    non_pruned_index_in = [{} for _ in range(args.pruning_steps)]

    pruned_weights_recorder = {}
    non_pruned_weights_recorder = {}

    
    # ====================ITERATIVE PRUNER================== #
    for i in range(args.pruning_steps):
        print(f"================PRUNING STAGE-{i+1}======================")
        for grp in pruner.step(interactive=True):
            # print(grp)
            # print("======================================")

            for dep, idxs in grp:
                layer = dep.target.module
                source_layer_name = dep.source._name
                target_layer_name = dep.target._name

                trigger = dep.trigger.__name__
                handler = dep.handler.__name__

                if isinstance(layer, (nn.Linear, nn.BatchNorm2d, nn.Conv2d)):

                    # if target_layer_name == "conv1":
                    #     print(target_layer_name)
                    #     print(source_layer_name)
                    #     print(type(layer))
                    #     print(handler)
                    #     print(trigger)
                    #     print(idxs)

                    if handler == "prune_out_channels":
                        for j in range(i):
                            if target_layer_name in pruned_index_out[j].keys() and set(pruned_index_out[j][target_layer_name]).issubset(set(idxs)):
                                idxs = [item for item in idxs if item not in pruned_index_out[j][target_layer_name]]

                        pruned_index_out[i][target_layer_name] = idxs
                        idxs_non = get_unpruned_indices(orig_dimensions[f"{target_layer_name}.weight"], idxs)
                        for j in range(i):
                            if target_layer_name in (non_pruned_index_out[j].keys() | pruned_index_out[j].keys()):
                                idxs_non = [item for item in idxs_non if item not in pruned_index_out[j][target_layer_name]]
                        non_pruned_index_out[i][target_layer_name] = idxs_non

                    elif handler == "prune_in_channels":
                        for j in range(i):
                            if target_layer_name in pruned_index_in[j].keys() and set(pruned_index_in[j][target_layer_name]).issubset(set(idxs)):
                                idxs = [item for item in idxs if item not in pruned_index_in[j][target_layer_name]]
                                
                        pruned_index_in[i][target_layer_name] = idxs
                        idxs_non = get_unpruned_indices(orig_dimensions[f"{target_layer_name}.weight"], idxs, dim="in")
                        for j in range(i):
                            if target_layer_name in (non_pruned_index_in[j].keys() | pruned_index_in[j].keys()):
                                idxs_non = [item for item in idxs_non if item not in pruned_index_in[j][target_layer_name]]
                        non_pruned_index_in[i][target_layer_name] = idxs_non

            if (i+1) == args.pruning_steps:
                grp.prune()

        # Saving Pruned metadata
        pruned_weights = extract_weight_subsets(orig_copy, pruned_index_out[i], pruned_index_in[i])
        pruned_weights_recorder[f"Level_{args.pruning_steps + 1 - i}"] = pruned_weights

        # Saving Non-Pruned metadata
        non_pruned_weights = extract_weight_subsets(orig_copy, non_pruned_index_out[i], non_pruned_index_in[i])
        non_pruned_weights_recorder[f"Level_{args.pruning_steps + 1 - i}"] = non_pruned_weights

        checkpoint = {"pruned_weights": pruned_weights,
                      "non_pruned_weights": non_pruned_weights,
                      "pruned_indexes": [pruned_index_in[args.pruning_steps-i-1], pruned_index_out[args.pruning_steps-i-1]],
                      "non_pruned_indexes": [non_pruned_index_in[args.pruning_steps-i-1], non_pruned_index_out[args.pruning_steps-i-1]]}
        # torch.save(checkpoint, f"./saves/pruning_metadata/{args.exp_name}_pruning_metadata_Level_{args.pruning_steps + 1 - i}.pth")

        print(f"Pruning Metadata stored for Level-{args.pruning_steps + 1 - i}")

    print("Iterative Pruning complete")

    del pruned_weights, non_pruned_weights

    
    ############################################################################################################
    # ---------------------------------------------FOR DEBUGGING-----------------------------------------------#
    
    # merged_out = []
    # merged_in = []
    # sample = 'conv1'

    # for i in range(args.pruning_steps):
    #     print(f"================PRUNING STAGE-{i+1}======================")
        # for key,value in pruned_index_out[args.pruning_steps-i-1].items():
        #     print("Out/Dim 0 Indices")
        #     print(key, value)
        #     print("# Pruned Index:", len(value))
        #     print("NON-PRUNED")
        #     print(non_pruned_index_out[args.pruning_steps-i-1][key])
        #     print("# Non-Pruned Index:", len(non_pruned_index_out[args.pruning_steps-i-1][key]))
        #     print(set(value) & set(non_pruned_index_out[args.pruning_steps-i-1][key]))
        # for key,value in pruned_index_in[args.pruning_steps-i-1].items():
        #     print("In/Dim 1 Indices")
        #     print(len(value))
        #     print(key, value)
        #     print("# Pruned Index:", len(value))
        #     print("NON-PRUNED")
        #     print(non_pruned_index_in[args.pruning_steps-i-1][key])
        #     print("# Non-Pruned Index:", len(non_pruned_index_in[args.pruning_steps-i-1][key]))
        #     print(set(value) & set(non_pruned_index_in[args.pruning_steps-i-1][key]))

    #     print(pruned_index_out[args.pruning_steps-i-1].keys())
    #     print(pruned_index_in[args.pruning_steps-i-1].keys())

    #     if sample in pruned_index_out[args.pruning_steps-i-1]:
    #         print("Pruned Out Indices")
    #         print(pruned_index_out[args.pruning_steps-i-1][sample])
    #         print("Number of Index Pruned:", len(pruned_index_out[args.pruning_steps-i-1][sample]))
    #         merged_out = merged_out + pruned_index_out[args.pruning_steps-i-1][sample]
    #     if sample in pruned_index_in[args.pruning_steps-i-1]:
    #         print("Pruned In Indices")
    #         print(pruned_index_in[args.pruning_steps-i-1][sample])
    #         print("Number of Index Pruned:", len(pruned_index_in[args.pruning_steps-i-1][sample]))
    #         merged_in = merged_in + pruned_index_in[args.pruning_steps-i-1][sample]
    #     if sample in pruned_weights_recorder[f"Level_{i+2}"]:
    #         print(pruned_weights_recorder[f"Level_{i+2}"][sample]["Weight"].shape)
    #     if sample in non_pruned_weights_recorder[f"Level_{i+2}"]:
    #         print(non_pruned_weights_recorder[f"Level_{i+2}"][sample]["Weight"].shape)


    # # Check for duplicates
    # unique_items = set(merged_out)
    # duplicates = [item for item in unique_items if merged_out.count(item) > 1]
    # if len(duplicates) > 0:
    #     print("Found Duplicates in pruned out indices:", duplicates)

    # unique_items = set(merged_in)
    # duplicates = [item for item in unique_items if merged_in.count(item) > 1]
    # if len(duplicates) > 0:
    #     print("Found Duplicates in pruned in indices:", duplicates)

    # exit()
    ############################################################################################################

    model_info = get_model_info_core(non_pruned_weights=model.state_dict())
    print("Model Info:", model_info)

    if args.stochastic_depth:
        inject_stochastic_depth_resnet(model)

    # Fine-Tune the Core model
    print("================FINE-TUNING CORE MODEL/DESCENDANT MODEL LEVEL-1======================")
    fine_tuner_core(args, device, model, train_loader, val_loader, train_sampler, val_sampler)

    """
    NOTE: Update the Core model weights after fine-tuning : 
    We update the non-pruned weights after pruning Level-2 model (Which is the core model)
    """
    updated_core_weights = extract_core_weights(model)
    # checkpoint = torch.load(f"./saves/pruning_metadata/{args.exp_name}_pruning_metadata_Level_2.pth")
    # checkpoint["non_pruned_weights"] = updated_core_weights
    # checkpoint["classifier"] = [model.classifier.weight.detach().clone(), model.classifier.bias.detach().clone()]
    # torch.save(checkpoint, f"./saves/pruning_metadata/{args.exp_name}_pruning_metadata_Level_2.pth")
    non_pruned_weights_recorder["Level_2"] = updated_core_weights
    print("Updated Pruning Metadata for Level-2")
    
    if args.test_accuracy:
        print("Testing accuracy of the pruned model...")
        acc_pruned, loss_pruned = evaluate(model, criterion, val_loader, device=device, dist=args.distributed, CNN=True)
        print("Accuracy: %.4f, Loss: %.4f"%(acc_pruned, loss_pruned))

    if args.save_as is not None:
        print("Saving the pruned model to %s..."%args.save_as)
        os.makedirs(os.path.dirname(args.save_as), exist_ok=True)
        torch.save(model.state_dict(), f"{args.save_as}Vit_b_16_Core_Level_1_state_dict_{args.exp_name}.pth")
        pruned_size = os.path.getsize(f"{args.save_as}Vit_b_16_Core_Level_1_state_dict_{args.exp_name}.pth") / (1024 * 1024)

    print("----------------------------------------")
    print("Summary:")
    pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
    print("Base MACs: %.2f G, Pruned MACs: %.2f G"%(base_macs/1e9, pruned_macs/1e9))
    print("Base Params: %.2f M, Pruned Params: %.2f M"%(base_params/1e6, pruned_params/1e6))
    if args.test_accuracy:
        print("Base Loss: %.4f, Pruned Loss: %.4f"%(loss_ori, loss_pruned))
        print("Base Accuracy: %.4f, Pruned Accuracy: %.4f"%(acc_ori, acc_pruned))

    
    if args.rebuild:
        print("----------------------------------------START REBUILDING----------------------------------------")
        macs_recorder = []
        param_recorder = []
        size_recorder = []
        acc_recorder = []

        for i in range(args.pruning_steps):
            print(f"================REBUILDING DESCENDANT MODEL LEVEL-{i+2}======================")

            rebuilding_index = [non_pruned_index_in[args.pruning_steps-i-1], non_pruned_index_out[args.pruning_steps-i-1]]
            pruned_index = [pruned_index_in[args.pruning_steps-i-1], pruned_index_out[args.pruning_steps-i-1]]
            
            if args.model_name.startswith('resnet'):
                rebuild_dim = get_model_info_resnet(pruned_index, rebuilding_index)
                rebuilt_model = resnet_generator(arch="resnet50", channel_dict=rebuild_dim, num_classes=num_classes).to(device)
            else:
                rebuild_dim = get_model_info_vgg(pruned_index, rebuilding_index)
                rebuilt_model = VGG_AnyDepth(rebuild_dim, num_classes=num_classes).to(device) 
            
            print("Model Info:", rebuild_dim)
            rebuilt_model,_,non_pruned_index_mapped = update_global_weights(rebuilt_model, pruned_index, rebuilding_index, 
                                                    pruned_weights_recorder[f"Level_{i+2}"], non_pruned_weights_recorder[f"Level_{i+2}"], device=device)
            
            # partial freezing of grads for freezing the core weights
            freeze_partial_weights_cnn(rebuilt_model, non_pruned_index_mapped[0], non_pruned_index_mapped[1], device)
            
            if args.stochastic_depth:
                inject_stochastic_depth_resnet(rebuilt_model)
            fine_tuner(args, device, rebuilt_model, train_loader, val_loader, train_sampler, val_sampler, rebuild=True, in_freeze_indices=non_pruned_index_mapped[0], out_freeze_indices=non_pruned_index_mapped[1])

            if i < args.pruning_steps - 1:
                updated_weights = extract_core_weights(rebuilt_model)
                # checkpoint = torch.load(f"./saves/pruning_metadata/{args.exp_name}_pruning_metadata_Level_{i+3}.pth")
                # checkpoint["non_pruned_weights"] = updated_weights
                # checkpoint["classifier"] = [rebuilt_model.classifier.weight.detach().clone(), rebuilt_model.classifier.bias.detach().clone()]
                # torch.save(checkpoint, f"./saves/pruning_metadata/{args.exp_name}_pruning_metadata_Level_{i+3}.pth")
                non_pruned_weights_recorder[f"Level_{i+3}"] = updated_weights
                print(f"Updated Pruning Metadata for Level-{i+3}")
                
                # DELETE UNUSED TENSORS TO FREE MEMORY
                del updated_weights

            if args.test_accuracy:
                print("Testing accuracy of the rebuild model...")
                acc_rebuilt, loss_rebuilt = evaluate(rebuilt_model, criterion, val_loader, device=device, dist=args.distributed, CNN=True)
                acc_recorder.append(acc_rebuilt)
                print("Accuracy: %.4f, Loss: %.4f"%(acc_rebuilt, loss_rebuilt))

            if args.save_as is not None: 
                print("Saving the rebuilt model to %s..."%args.save_as)
                torch.save(rebuilt_model.state_dict(), f"{args.save_as}Vit_b_16_Rebuilt_Level_{i+2}_state_dict_{args.exp_name}.pth")
                rebuilt_size = os.path.getsize(f"{args.save_as}Vit_b_16_Rebuilt_Level_{i+2}_state_dict_{args.exp_name}.pth") / (1024 * 1024)
                size_recorder.append(rebuilt_size)

            print("----------------------------------------")
            print(f"Summary Level-{i+2}:")
            rebuilt_macs, rebuilt_params = tp.utils.count_ops_and_params(rebuilt_model, example_inputs)
            macs_recorder.append(rebuilt_macs)
            param_recorder.append(rebuilt_params)
            print("Base MACs: %.2f G, Pruned MACs: %.2f G, Rebuilt MACs: %.2f G"%(base_macs/1e9, pruned_macs/1e9, rebuilt_macs/1e9))
            print("Base Params: %.2f M, Pruned Params: %.2f M, Rebuilt Params: %.2f M"%(base_params/1e6, pruned_params/1e6, rebuilt_params/1e6))
            if args.test_accuracy:
                print("Base Loss: %.4f, Pruned Loss: %.4f, Rebuilt Loss: %.4f"%(loss_ori, loss_pruned, loss_rebuilt))
                print("Base Accuracy: %.4f, Pruned Accuracy: %.4f, Rebuilt Accuracy: %.4f"%(acc_ori, acc_pruned, acc_rebuilt))

            # DELETE UNUSED TENSORS TO FREE MEMORY
            del pruned_index, rebuilding_index, rebuilt_model
            torch.cuda.empty_cache()

            plot_comparison_macs(args, accuracy=[acc_ori, acc_pruned, *acc_recorder], macs=[base_macs, pruned_macs, *macs_recorder], x_labels=(["Original"] + [f"Level-{j+1}" for j in range(i+2)]))
            plot_comparison_params(args, accuracy=[acc_ori, acc_pruned, *acc_recorder], params=[base_params, pruned_params, *param_recorder], x_labels=(["Original"] + [f"Level-{j+1}" for j in range(i+2)]))
            plot_comparison_size(args, accuracy=[acc_ori, acc_pruned, *acc_recorder], size=[base_size, pruned_size, *size_recorder], x_labels=(["Original"] + [f"Level-{j+1}" for j in range(i+2)]))
            print("Test Accuracy:", [acc_ori, acc_pruned, *acc_recorder])
            print("MAC's:", [base_macs, pruned_macs, *macs_recorder])
            print("Params:", [base_params, pruned_params, *param_recorder])
            print("Size:", [base_size, pruned_size, *size_recorder])
            print("Plotting Complete")


        
        
        plot_comparison_macs(args, accuracy=[acc_ori, acc_pruned, *acc_recorder], macs=[base_macs, pruned_macs, *macs_recorder], x_labels=(["Original"] + [f"Level-{i}" for i in range(1, args.pruning_steps+2)]))
        plot_comparison_params(args, accuracy=[acc_ori, acc_pruned, *acc_recorder], params=[base_params, pruned_params, *param_recorder], x_labels=(["Original"] + [f"Level-{i}" for i in range(1, args.pruning_steps+2)]))
        plot_comparison_size(args, accuracy=[acc_ori, acc_pruned, *acc_recorder], size=[base_size, pruned_size, *size_recorder], x_labels=(["Original"] + [f"Level-{i}" for i in range(1, args.pruning_steps+2)]))
        print("Test Accuracy:", [acc_ori, acc_pruned, *acc_recorder])
        print("MAC's:", [base_macs, pruned_macs, *macs_recorder])
        print("Params:", [base_params, pruned_params, *param_recorder])
        print("Size:", [base_size, pruned_size, *size_recorder])
        print("Rebuilding Complete")
    


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
