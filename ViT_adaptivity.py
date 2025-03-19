import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
from typing import Dict, List, Optional, Set, Tuple, Union
import torch
import torch.nn.functional as F
import torch_pruning as tp
from transformers import ViTForImageClassification, ViTModel, ViTConfig
from transformers.models.vit.modeling_vit import ViTSelfAttention, ViTSelfOutput, ViTLayer
import transformers
import warnings
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import torchvision.models as models
from torch.utils.data import SubsetRandomSampler, Subset
from tqdm import tqdm
import imgaug_presets
from copy import deepcopy
# from models.hf_vit import create_vit_general
from prune_utils import *
from trainer import fine_tuner, evaluate
from sampler import RASampler
import utils

import argparse


def get_args_parser(add_help=True):

    parser = argparse.ArgumentParser(description="ViT Pruning and PyTorch Classification Training", add_help=add_help)

    ###################################################################################################################
    #                                                   PRUNING                                                       #
    ###################################################################################################################

    parser.add_argument('--exp_name', default='ViT_Adaptivity', type=str, help='Name of the experiment')
    parser.add_argument('--model_name', default='google/vit-base-patch16-224', type=str, help='model name')
    parser.add_argument('--data_path', default='data/imagenet', type=str, help='model name')
    parser.add_argument('--taylor_batchs', default=10, type=int, help='number of batchs for taylor criterion')
    parser.add_argument('--pruning_ratio', default=0.5, type=float, help='prune ratio')
    parser.add_argument('--bottleneck', default=False, action='store_true', help='bottleneck or uniform')
    parser.add_argument('--pruning_type', default='l1', type=str, help='pruning type', choices=['random', 'taylor', 'l1'])
    parser.add_argument('--test_accuracy', default=False, action='store_true', help='test accuracy')
    parser.add_argument('--global_pruning', default=False, action='store_true', help='global pruning')
    parser.add_argument('--rebuild', default=False, action='store_true', help='Rebuilding for adaptivity')

    parser.add_argument('--train_batch_size', default=64, type=int, help='train batch size')
    parser.add_argument('--val_batch_size', default=128, type=int, help='val batch size')
    parser.add_argument('--save_as', default=None, type=str, help='save as')

    ###################################################################################################################
    #                                                 FINE-TUNING                                                     #
    ###################################################################################################################
    parser.add_argument("--data-path", default="/ssdstore/ImageNet/", type=str, help="dataset path")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=512, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=50, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="adamw", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.003, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=0.3,
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
    parser.add_argument("--lr-warmup-epochs", default=30, type=int, help="the number of epochs to warmup (default: 0)")
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
    parser.add_argument("--auto-augment", default="ra", type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

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
    parser.add_argument("--clip-grad-norm", default=1, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")
    parser.add_argument("--wandb", action="store_true", help="Use Weights and Bias to log the training curves")
    return parser


def prepare_imagenet(imagenet_root, train_batch_size=64, val_batch_size=128, num_workers=4, use_imagenet_mean_std=True, debug=False):
    """The imagenet_root should contain train and val folders.
    """

    print('Parsing dataset...')
    # train_dst = ImageFolder(os.path.join(imagenet_root, 'train'), 
    #                         transform=imgaug_presets.ClassificationPresetEval(
    #                             mean=[0.485, 0.456, 0.406] if use_imagenet_mean_std else [0.5, 0.5, 0.5],
    #                             std=[0.229, 0.224, 0.225] if use_imagenet_mean_std else [0.5, 0.5, 0.5],
    #                             crop_size=224,
    #                             resize_size=256,
    #                             interpolation=InterpolationMode.BILINEAR,
    #                         )
    # )

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

    # train_loader = torch.utils.data.DataLoader(train_dst, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    # val_loader = torch.utils.data.DataLoader(val_dst, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    # Just For Debugging
    if debug:
        train_dst = Subset(train_dst, indices=torch.randperm(len(train_dst))[:500])
        val_dst = Subset(val_dst, indices=torch.randperm(len(val_dst))[:100])
    else:
        train_dst = Subset(train_dst, indices=torch.randperm(len(train_dst))[:100000])
        val_dst = Subset(val_dst, indices=torch.randperm(len(val_dst))[:10000])

    
    if args.distributed:    
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(train_dst, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dst)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dst, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dst)
        val_sampler = torch.utils.data.SequentialSampler(val_dst)


    train_loader = torch.utils.data.DataLoader(train_dst, batch_size=val_batch_size, sampler=train_sampler, shuffle=False, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dst, batch_size=val_batch_size, sampler=val_sampler, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, train_sampler, val_sampler

def validate_model(model, val_loader, device, hf=True):
    model.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            if hf:
                outputs = model(images).logits
            else:
                outputs = model(images)
            loss += torch.nn.functional.cross_entropy(outputs, labels, reduction='sum').item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    return correct / len(val_loader.dataset), loss / len(val_loader.dataset)

def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )

        attention_output = self_attention_outputs[0]
        print("attention outputs: ", attention_output.shape)
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        print("FF1 outputs: ", layer_output.shape)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)
        print("FF2 outputs: ", layer_output.shape)

        outputs = (layer_output,) + outputs

        return outputs

def create_vit_general(img_size=(224,224), patch_size=(16,16), in_channels=3, embed_dim=768, num_layers=12, num_heads=12, 
                  qkv_dim=768, ff_hidden_dim=3072, output_dim=768, num_classes=1000, dim_dict=None):
    """
    Creates a Hugging Face Vision Transformer (ViT) model with customizable dimensions.

    Args:
        img_size (int): Image size (default: 224x224).
        patch_size (int): Patch size for embedding.
        in_channels (int): Number of image channels (default: 3).
        embed_dim (int): Dimension of embedding layer.
        num_layers (int): Number of transformer blocks.
        num_heads (int): Number of attention heads.
        qkv_dim (int): Dimension of Query, Key, and Value weights.
        ff_hidden_dim (int): Hidden dimension in feed-forward layers.
        output_dim (int): Output dimension of FFN.
        num_classes (int): Number of output classes.

    Returns:
        ViTModel: A Hugging Face ViT model with custom configurations.
    """

    if dim_dict is not None:
        embed_dim = dim_dict['vit.embeddings.patch_embeddings.projection.weight']
        num_layers = max(int(key.split(".")[3]) for key in dim_dict if key.startswith("vit.encoder.layer")) + 1
        num_heads = dim_dict['num_heads']
        ff_hidden_dim = dim_dict['vit.encoder.layer.0.intermediate.dense.weight'][0]
        qkv_dim = embed_dim
        output_dim = dim_dict['vit.encoder.layer.0.output.dense.weight'][0]

    config = ViTConfig(
        image_size=img_size,
        patch_size=patch_size,
        num_channels=in_channels,
        hidden_size=embed_dim,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=ff_hidden_dim,
        qkv_bias=True,  # Include biases for Q, K, V
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
        num_labels=num_classes,
    )

    model = ViTForImageClassification(config)
    return model

def compare_performance(args):
    device = torch.device(args.device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    path_core = f"./saves/state_dicts/Vit_b_16_Pruned_{args.pruning_ratio}_state_dict.pth"
    path_rebuilt = f"./saves/state_dicts/Vit_b_16_Rebuilt_{args.pruning_ratio}_state_dict.pth"

    prune_dict = torch.load(path_core, map_location=device)
    rebuilt_dict = torch.load(path_rebuilt, map_location=device)

    print(prune_dict.keys())
    print(prune_dict['vit.encoder.layer.0.attention.attention.query.weight'].shape)
    print(prune_dict['vit.encoder.layer.0.intermediate.dense.weight'].shape)

    prune_embed = prune_dict['vit.encoder.layer.0.attention.attention.query.weight'].shape[0]
    prune_ff = prune_dict['vit.encoder.layer.0.intermediate.dense.weight'].shape[0]

    rebuilt_embed = rebuilt_dict['vit.encoder.layer.0.attention.attention.query.weight'].shape[0]
    rebuilt_ff = rebuilt_dict['vit.encoder.layer.0.intermediate.dense.weight'].shape[0]

    example_inputs = torch.randn(1,3,224,224).to(device)

    _,val_loader,_,_ = prepare_imagenet(args.data_path, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size)

    original_model = ViTForImageClassification.from_pretrained(args.model_name).to(device)
    pruned_model = create_vit_general(embed_dim=prune_embed, output_dim=prune_embed, ff_hidden_dim=prune_ff).to(device)
    rebuilt_model = create_vit_general(embed_dim=rebuilt_embed, output_dim=rebuilt_embed, ff_hidden_dim=rebuilt_ff).to(device)

    pruned_model.load_state_dict(torch.load(path_core, map_location=device))
    rebuilt_model.load_state_dict(torch.load(path_rebuilt, map_location=device))

    base_macs,_ = tp.utils.count_ops_and_params(original_model, example_inputs)
    pruned_macs,_ = tp.utils.count_ops_and_params(pruned_model, example_inputs)
    rebuilt_macs,_ = tp.utils.count_ops_and_params(rebuilt_model, example_inputs)

    orig_acc = evaluate(original_model, criterion, val_loader, device=device)
    pruned_acc = evaluate(pruned_model, criterion, val_loader, device=device)
    rebuilt_acc = evaluate(rebuilt_model, criterion, val_loader, device=device)

    plot_comparison(accuracy=[orig_acc, pruned_acc, rebuilt_acc], macs=[base_macs, pruned_macs, rebuilt_macs], pruning_ratio=args.pruning_ratio)



def main(args):

    if args.distributed:
            utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    example_inputs = torch.randn(1,3,224,224).to(device)

    if args.pruning_type == 'random':
        imp = tp.importance.RandomImportance()
    elif args.pruning_type == 'taylor':
        imp = tp.importance.GroupTaylorImportance()
    elif args.pruning_type == 'l2':
        imp = tp.importance.GroupNormImportance(p=2)
    elif args.pruning_type == 'l1':
        imp = tp.importance.GroupNormImportance(p=1)
    elif args.pruning_type == 'hessian':
        imp = tp.importance.GroupHessianImportance()
    else: raise NotImplementedError

    if args.pruning_type=='taylor' or args.test_accuracy:
        train_loader, val_loader, train_sampler, val_sampler = prepare_imagenet(args.data_path, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size, debug=True)


    # Load the model
    model = ViTForImageClassification.from_pretrained(args.model_name).to(device)
    # model = create_vit_general().to(device)
    # model = models.resnet18(pretrained=True).to(device)
    orig_copy = deepcopy(model)
    # torch.onnx.export(model, (example_inputs,), f=f"./saves/onnx/ViT_b_16.onnx")
    # for name, layer in model.named_parameters():
    #     if layer.requires_grad:
    #         print(name)
    #         print(type(layer))
    # exit()


    orig_statedict = orig_copy.state_dict()
    orig_dimensions = get_layer_size(orig_statedict)
    orig_dimensions['num_heads'] = orig_copy.config.num_attention_heads

    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"Orig MAC's: {base_macs/1e9:.2f} G, Orig Params: {base_params/1e6:.2f} M")

    # visualize_weight_matrix(model, "vit.encoder.layer.0.attention.attention.query.weight", plot_name="original")

    if args.test_accuracy:
        print("Testing accuracy of the original model...")
        acc_ori, loss_ori = validate_model(model, val_loader, device)
        print("Accuracy: %.4f, Loss: %.4f"%(acc_ori, loss_ori))

    print("Pruning %s..."%args.model_name)
    num_heads = {}
    # ignored_layers = [model.vit.embeddings, model.classifier]
    ignored_layers = [model.classifier]
    # All heads should be pruned simultaneously, so we group channels by head.
    for m in model.modules():
        # if isinstance(m, ViTLayer):
        #     m.forward = forward.__get__(m, transformers.models.vit.modeling_vit.ViTLayer)
        if isinstance(m, ViTSelfAttention):
            num_heads[m.query] = m.num_attention_heads
            num_heads[m.key] = m.num_attention_heads
            num_heads[m.value] = m.num_attention_heads
        if args.bottleneck and isinstance(m, ViTSelfOutput):
            ignored_layers.append(m.dense) # only prune the internal layers of FFN & Attention
                
    
    # pruner = tp.pruner.MetaPruner(
    #             model, 
    #             example_inputs, 
    #             global_pruning=args.global_pruning, # If False, a uniform pruning ratio will be assigned to different layers.
    #             importance=imp, # importance criterion for parameter selection
    #             pruning_ratio=args.pruning_ratio, # target pruning ratio
    #             ignored_layers=ignored_layers,
    #             round_to=1,
    #             num_heads=num_heads,
    #             prune_head_dims=True,
    #             prune_num_heads=False,                 # Removes entire heads (Contigous columns of indices)
    #             head_pruning_ratio=args.pruning_ratio, # disabled when prune_num_heads=False
    #             output_transform=lambda out: out.logits.sum() # Transform to convert logits to scalar
    # )

    pruner = tp.pruner.MetaPruner(
                model, 
                example_inputs, 
                global_pruning=args.global_pruning, # If False, a uniform pruning ratio will be assigned to different layers.
                importance=imp, # importance criterion for parameter selection
                pruning_ratio=args.pruning_ratio, # target pruning ratio
                ignored_layers=ignored_layers,
                round_to=1,
                num_heads=num_heads,
                prune_head_dims=True,
                output_transform=lambda out: out.logits.sum() # Transform to convert logits to scalar
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

    pruned_index_out = {}
    pruned_index_in = {}

    non_pruned_index_out = {}
    non_pruned_index_in = {}

    for grp in pruner.step(interactive=True):
        # print(grp)
        # print("======================================")

        for g in grp:
            dep, idxs = g
            layer = dep.target.module
            source_layer_name = dep.source._name
            target_layer_name = dep.target._name

            trigger = dep.trigger.__name__
            handler = dep.handler.__name__

            if isinstance(layer, (nn.Linear, nn.LayerNorm, nn.Conv2d)):

                # print(target_layer_name)
                # print(source_layer_name)
                # print(type(layer))
                # print(handler)
                # print(trigger)
                # print(idxs)

                # if isinstance(layer, nn.Conv2d):
                #     print(target_layer_name)
                #     print("Output channels Pruned: ", len(idxs))

                if handler == "prune_out_channels":
                    pruned_index_out[target_layer_name] = idxs
                    non_pruned_index_out[target_layer_name] = get_unpruned_indices(orig_dimensions[f"{target_layer_name}.weight"], idxs)

                    # print(pruned_index_out[target_layer_name])
                    # print(non_pruned_index_out[target_layer_name])
                    # print("--------------------------------------")
                elif handler == "prune_in_channels":
                    pruned_index_in[target_layer_name] = idxs
                    non_pruned_index_in[target_layer_name] = get_unpruned_indices(orig_dimensions[f"{target_layer_name}.weight"], idxs, dim="in")

                    # print(pruned_index_in[target_layer_name])
                    # print(non_pruned_index_in[target_layer_name])
                    # print("--------------------------------------")
    
        grp.prune()


    ############################################################################################################
    # ---------------------------------------------FOR DEBUGGING-----------------------------------------------#

    # print(pruned_index_out.keys())
    # print(pruned_index_in.keys())

    # common_keys = (sorted(list(pruned_index_in.keys() & pruned_index_out.keys()), key=extract_layer_index))
    # unique_dim_out = pruned_index_out.keys() - pruned_index_in.keys()
    # unique_dim_in = pruned_index_in.keys() - pruned_index_out.keys()
    # print(sorted(list(common_keys)))
    # print(unique_dim_out)
    # print(unique_dim_in)

    # for key in common_keys:
    #     print(f"Pruned out Indices {key} => {pruned_index_out[key]}")
    #     print(len(pruned_index_out[key]))
    #     print(f"Pruned in Indices {key} => {pruned_index_in[key]}")
    #     print(len(pruned_index_in[key]))
    #     print()
    ############################################################################################################

    pruned_weights = extract_vit_weight_subset(orig_copy, pruned_index_out, pruned_index_in)
    with open(f"./saves/pruning_metadata/pruned_ViT_weights.txt", "w") as file:
        for key, value in pruned_weights.items():
            file.write(f"  {key}: {value}\n")

    non_pruned_weights = extract_vit_weight_subset(orig_copy, non_pruned_index_out, non_pruned_index_in)
    with open(f"./saves/pruning_metadata/non_pruned_ViT_weights.txt", "w") as file:
        for key, value in non_pruned_weights.items():
            file.write(f"  {key}: {value}\n")

    # Modify the attention head size and all head size after pruning
    for id, m in enumerate(model.modules()):
        if isinstance(m, ViTSelfAttention):
            print("Layer:", id)
            print("num_heads:", m.num_attention_heads, 'head_dims:', m.attention_head_size, 'all_head_size:', m.all_head_size, '=>')
            m.num_attention_heads = pruner.num_heads[m.query]
            m.attention_head_size = m.query.out_features // m.num_attention_heads
            m.all_head_size = m.query.out_features
            print("num_heads:", m.num_attention_heads, 'head_dims:', m.attention_head_size, 'all_head_size:', m.all_head_size)
            print()

    # Fine-Tune the pruned model
    # fine_tuner(args, model, train_loader, val_loader, train_sampler, val_sampler)

    if args.test_accuracy:
        print("Testing accuracy of the pruned model...")
        acc_pruned, loss_pruned = validate_model(model, val_loader, device)
        print("Accuracy: %.4f, Loss: %.4f"%(acc_pruned, loss_pruned))

    if args.save_as is not None:
        print("Saving the pruned model to %s..."%args.save_as)
        os.makedirs(os.path.dirname(args.save_as), exist_ok=True)
        torch.save(model.state_dict(), f"{args.save_as}Vit_b_16_Pruned_{args.pruning_ratio}_state_dict_{args.exp_name}.pth")

    print("----------------------------------------")
    print("Summary:")
    pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
    print("Base MACs: %.2f G, Pruned MACs: %.2f G"%(base_macs/1e9, pruned_macs/1e9))
    print("Base Params: %.2f M, Pruned Params: %.2f M"%(base_params/1e6, pruned_params/1e6))
    if args.test_accuracy:
        print("Base Loss: %.4f, Pruned Loss: %.4f"%(loss_ori, loss_pruned))
        print("Base Accuracy: %.4f, Pruned Accuracy: %.4f"%(acc_ori, acc_pruned))

    # visualize_weight_matrix(model, "vit.encoder.layer.0.attention.attention.query.weight", plot_name="pruned")



    if args.rebuild:
        rebuilt_model = create_vit_general(dim_dict=orig_dimensions).to(device)
        # rebuilt_model = ViTForImageClassification.from_pretrained(args.model_name).to(device)

        rebuilt_model = update_vit_weights(rebuilt_model, [pruned_index_in, pruned_index_out], [non_pruned_index_in, 
                                        non_pruned_index_out], pruned_weights, non_pruned_weights).to(device)

        # partial freezing of grads for freezing the core weights
        freeze_partial_weights(rebuilt_model, pruned_index_in, pruned_index_out, device)
        exit()

        fine_tuner(args, rebuilt_model, train_loader, val_loader, train_sampler, val_sampler)
        
        if args.test_accuracy:
            print("Testing accuracy of the rebuild model...")
            acc_rebuilt, loss_rebuilt = validate_model(rebuilt_model, val_loader, device)
            print("Accuracy: %.4f, Loss: %.4f"%(acc_rebuilt, loss_rebuilt))

        if args.save_as is not None: 
            print("Saving the rebuilt model to %s..."%args.save_as)
            torch.save(rebuilt_model.state_dict(), f"{args.save_as}Vit_b_16_Rebuilt_{args.pruning_ratio}_state_dict_{args.exp_name}.pth")

        print("----------------------------------------")
        print("Summary:")
        rebuilt_macs, rebuilt_params = tp.utils.count_ops_and_params(rebuilt_model, example_inputs)
        print("Base MACs: %.2f G, Pruned MACs: %.2f G, Rebuilt MACs: %.2f G"%(base_macs/1e9, pruned_macs/1e9, rebuilt_macs/1e9))
        print("Base Params: %.2f M, Pruned Params: %.2f M, Rebuilt Params: %.2f M"%(base_params/1e6, pruned_params/1e6, rebuilt_params/1e6))
        if args.test_accuracy:
            print("Base Loss: %.4f, Pruned Loss: %.4f, Rebuilt Loss: %.4f"%(loss_ori, loss_pruned, loss_rebuilt))
            print("Base Accuracy: %.4f, Pruned Accuracy: %.4f, Rebuilt Accuracy: %.4f"%(acc_ori, acc_pruned, acc_rebuilt))


    


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)

    # compare_performance(args)
