import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import torch
import torch_pruning as tp
from tqdm import tqdm
from copy import deepcopy
from trainer import fine_tuner, evaluate, fine_tuner_core
import utils.train_utils as train_utils
import argparse
from datasets import load_imagenet, load_cifar, load_imagenette, load_dummy_data
from models.elastoformer import *
from utils.prune_utils import *

torch.manual_seed(42)

def get_args_parser(add_help=True):

    parser = argparse.ArgumentParser(description="ViT Pruning and PyTorch Classification Training", add_help=add_help)

    ###################################################################################################################
    #                                                   PRUNING                                                       #
    ###################################################################################################################

    parser.add_argument('--exp_name', default='ViT_Adaptivity', type=str, help='Name of the experiment')
    parser.add_argument('--model_name', default='google/vit-base-patch16-224', type=str, help='model name')
    parser.add_argument('--dataset_name', default='imagenet', type=str, help='Dataset used')
    parser.add_argument('--data_path', default='data/imagenet', type=str, help='model name')
    parser.add_argument('--taylor_batchs', default=10, type=int, help='number of batchs for taylor criterion')
    parser.add_argument('--pruning_ratio', default=0.5, type=float, help='prune ratio')
    parser.add_argument('--iterative', default=True, action='store_true', help='True for Iterative Pruning')
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
    parser.add_argument("--output-dir", default="/var/scratch/skalra/elastoformer_saves/Checkpoints/", type=str, help="path to save outputs")
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


def main(args):

    if args.distributed:
            train_utils.init_distributed_mode(args)
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    example_inputs = torch.randn(1,3,224,224).to(device)

    # PRUNING CRITERION
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

    # DATALOADERS
    if args.dataset_name.startswith('imagenet'):
        train_loader, val_loader, num_classes = load_imagenet(datapath=args.data_path, batch_size=args.train_batch_size, distributed=args.distributed, ra_sampler=args.ra_sampler, debug=args.debug)
    if args.dataset_name.startswith('imagenette'):
        train_loader, val_loader, train_sampler, val_sampler = load_imagenette(args)
    if args.dataset_name.startswith('cifar'):
        train_loader, val_loader, train_sampler, val_sampler, num_classes = load_cifar(dataset=args.dataset_name, batch_size=args.train_batch_size, distributed=args.distributed)
    else:
        train_loader, val_loader, num_classes = load_dummy_data(batch_size=args.train_batch_size, distributed=args.distributed)
    
    
    # MODEL INITIALIZATION
    if "small" in args.model_name:
        model = ElasticViTForImageClassification.from_pretrained(args.model_name, pruned_dim=384)
    else:
        model = ElasticViTForImageClassification.from_pretrained(args.model_name)
    if args.dataset_name.startswith('cifar'):
        model.classifier = nn.Linear(model.config.hidden_size, num_classes)
    model = model.to(device)
    print("Model loaded:", args.model_name)

    # Fine-Tuning classifier (IF REQUIRED, ONLY FOR IMAGENETTE)
    if args.dataset_name == "imagenette" and args.rebuild:
        print("Fine-tuning classifier on Imagenette...")
        for param in model.vit.parameters():
            param.requires_grad = False

        print_trainable_summary(model)
        for name, param in model.named_parameters():
            print(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'} - {param.numel():,} params")

        if args.test_accuracy:
            criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
            print("Testing accuracy of the original model...")
            acc_ori, loss_ori = evaluate(model, criterion, val_loader, device=device, dist=args.distributed)
            print("Accuracy: %.4f, Loss: %.4f"%(acc_ori, loss_ori))

        fine_tuner(args, device, model, train_loader, val_loader)
        print("Fine-tuning complete")
        if args.test_accuracy:
            criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
            print("Testing accuracy of the original model...")
            acc_ori, loss_ori = evaluate(model, criterion, val_loader, device=device, dist=args.distributed)
            print("Accuracy: %.4f, Loss: %.4f"%(acc_ori, loss_ori))
        
        model.save_pretrained("./pretrained_weights/vit/vit_base_patch16_224_ft_in1k")

    
    orig_copy = deepcopy(model)

    orig_statedict = orig_copy.state_dict()
    orig_dimensions = get_layer_size(orig_statedict)
    orig_dimensions['num_heads'] = orig_copy.config.num_attention_heads

    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
    if args.distributed:
        base_size = get_model_size_mb_multi(model)
    else:
        base_size = get_model_size_mb(model)
    print(f"Orig MAC's: {base_macs/1e9:.2f} G, Orig Params: {base_params/1e6:.2f} M, Orig Size: {base_size:.2f} MB")
    
    if args.test_accuracy:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        print("Testing accuracy of the original model...")
        acc_ori, loss_ori = evaluate(orig_copy, criterion, val_loader, device=device, dist=args.distributed)
        print("Accuracy: %.4f %%, Loss: %.4f" % (acc_ori, loss_ori))
    
    print("Pruning %s..."%args.model_name)
    num_heads = {}
    ignored_layers = [model.classifier]
    # All heads should be pruned simultaneously, so we group channels by head.
    for m in model.modules():
        if isinstance(m, ElasticViTSelfAttention):
            num_heads[m.query] = m.num_attention_heads
            num_heads[m.key] = m.num_attention_heads
            num_heads[m.value] = m.num_attention_heads
        if args.bottleneck and isinstance(m, ElasticViTSelfOutput):
            ignored_layers.append(m.dense) # only prune the internal layers of FFN & Attention
                
    pruner = tp.pruner.BasePruner(
                model, 
                example_inputs,
                isomorphic=args.isomorphic,
                global_pruning=args.global_pruning, # If False, a uniform pruning ratio will be assigned to different layers.
                importance=imp, # importance criterion for parameter selection
                iterative_steps=args.pruning_steps, # number of pruning steps
                pruning_ratio=args.pruning_ratio, # target pruning ratio
                ignored_layers=ignored_layers,
                round_to=1,
                num_heads=num_heads,
                prune_num_heads = False,  # remove entire heads in multi-head attention. Default: False.
                prune_head_dims = True,   # remove head dimensions in multi-head attention. Default: True.
                head_pruning_ratio = args.pruning_ratio, # head pruning ratio. Default: 0.0.
                head_pruning_ratio_dict = None, # (Dict[nn.Module, float]): layer-specific head pruning ratio. Default: None.
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

    # Pruning Metadata for Rebuilding
    pruned_index_out = [{} for _ in range(args.pruning_steps)]
    pruned_index_in = [{} for _ in range(args.pruning_steps)]

    non_pruned_index_out = [{} for _ in range(args.pruning_steps)]
    non_pruned_index_in = [{} for _ in range(args.pruning_steps)]

    pruned_weights_recorder = {}
    non_pruned_weights_recorder = {}
    
    # Convert index lists → tensors
    def dict_to_tensor(d):
        return {k: torch.tensor(v, dtype=torch.int64) for k, v in d.items()}
    
    def tensorize_weight_dict(subdict):
        out = {}
        for name, val in subdict.items():
            if val is None:
                out[name] = None
            else:
                out[name] = (
                    val if isinstance(val, torch.Tensor) else torch.tensor(val)
                ).cpu()
        return out

    
    ## ====================ITERATIVE PRUNING================== ##

    for i in range(args.pruning_steps):
        print(f"================PRUNING STAGE-{i+1}======================")
        for grp in pruner.step(interactive=True):

            for dep, idxs in grp:
                layer = dep.target.module
                source_layer_name = dep.source._name
                target_layer_name = dep.target._name

                trigger = dep.trigger.__name__
                handler = dep.handler.__name__

                if isinstance(layer, (nn.Linear, nn.LayerNorm, nn.Conv2d)):

                    if handler == "prune_out_channels":
                        for j in range(i):
                            if target_layer_name in pruned_index_out[j].keys() and set(pruned_index_out[j][target_layer_name]).issubset(set(idxs)):
                                idxs = [item for item in idxs if item not in pruned_index_out[j][target_layer_name]]

                        pruned_index_out[i][target_layer_name] = idxs
                        # non_pruned_index_out[i][target_layer_name] = get_unpruned_indices(orig_dimensions[f"{target_layer_name}.weight"], idxs)
                        idxs_non = get_unpruned_indices(orig_dimensions[f"{target_layer_name}.weight"], idxs)
                        for j in range(i):
                            if target_layer_name in (non_pruned_index_out[j].keys() | pruned_index_out[j].keys()):
                                idxs_non = [item for item in idxs_non if item not in pruned_index_out[j][target_layer_name]]
                        non_pruned_index_out[i][target_layer_name] = idxs_non
   

                    elif handler == "prune_in_channels":
                        for j in range(i):
                            if target_layer_name in pruned_index_in[j].keys() and set(pruned_index_in[j][target_layer_name]).issubset(set(idxs)):
                                idxs = [item for item in idxs if item not in pruned_index_in[j][target_layer_name]]
                                
                        # if i==0:
                        #     pruned_index_in[i][target_layer_name] = idxs
                        # else:
                        #     if target_layer_name in pruned_index_in[i-1].keys() and set(pruned_index_in[i-1][target_layer_name]).issubset(set(idxs)):
                        #         idxs = [item for item in idxs if item not in pruned_index_in[i-1][target_layer_name]]
                        pruned_index_in[i][target_layer_name] = idxs
                        # non_pruned_index_in[i][target_layer_name] = get_unpruned_indices(orig_dimensions[f"{target_layer_name}.weight"], idxs, dim="in")
                        idxs_non = get_unpruned_indices(orig_dimensions[f"{target_layer_name}.weight"], idxs, dim="in")
                        for j in range(i):
                            if target_layer_name in (non_pruned_index_in[j].keys() | pruned_index_in[j].keys()):
                                idxs_non = [item for item in idxs_non if item not in pruned_index_in[j][target_layer_name]]
                        non_pruned_index_in[i][target_layer_name] = idxs_non
        
            if (i+1) == args.pruning_steps:
                grp.prune()

        # Saving Pruned metadata
        pruned_weights = extract_vit_weight_subset(orig_copy, pruned_index_out[i], pruned_index_in[i])
        pruned_weights_recorder[f"Level_{args.pruning_steps + 1 - i}"] = pruned_weights

        # Saving Non-Pruned metadata
        non_pruned_weights = extract_vit_weight_subset(orig_copy, non_pruned_index_out[i], non_pruned_index_in[i])
        non_pruned_weights_recorder[f"Level_{args.pruning_steps + 1 - i}"] = non_pruned_weights

        # Save optimized Metadata for Adaptive Switching
        # metadata = {"pruned_weights": pruned_weights,
        #               "non_pruned_weights": non_pruned_weights,
        #               "pruned_indexes": [pruned_index_in[args.pruning_steps-i-1], pruned_index_out[args.pruning_steps-i-1]],
        #               "non_pruned_indexes": [non_pruned_index_in[args.pruning_steps-i-1], non_pruned_index_out[args.pruning_steps-i-1]]}
        # torch.save(metadata, f"/var/scratch/skalra/elastoformer_saves/pruning_metadata/{args.exp_name}_pruning_metadata_Level_{args.pruning_steps + 1 - i}.pth")

        # metadata = {
        #     "pruned_weights": {
        #         k: v.cpu() for k, v in pruned_weights.items()
        #     },
        #     "non_pruned_weights": {
        #         k: v.cpu() for k, v in non_pruned_weights.items()
        #     },

        #     "pruned_index_in": dict_to_tensor(pruned_index_in[args.pruning_steps-i-1]),
        #     "pruned_index_out": dict_to_tensor(pruned_index_out[args.pruning_steps-i-1]),

        #     "non_pruned_index_in": dict_to_tensor(non_pruned_index_in[args.pruning_steps-i-1]),
        #     "non_pruned_index_out": dict_to_tensor(non_pruned_index_out[args.pruning_steps-i-1]),
        # }

        # torch.save(metadata,
        #         f"./saves/pruning_metadata/{args.exp_name}_pruning_metadata_Level_{args.pruning_steps + 1 - i}.pth")

        save_path = f"./saves/pruning_metadata/{args.exp_name}_pruning_metadata_Level_{args.pruning_steps + 1 - i}.pth"
        
        checkpoint = {
            "pruned_weights": {
                layer: tensorize_weight_dict(wdict)
                for layer, wdict in pruned_weights.items()
            },
            "non_pruned_weights": {
                layer: tensorize_weight_dict(wdict)
                for layer, wdict in non_pruned_weights.items()
            },
            "pruned_index_in": dict_to_tensor(pruned_index_in[i]),
            "pruned_index_out": dict_to_tensor(pruned_index_out[i]),
            "non_pruned_index_in": dict_to_tensor(non_pruned_index_in[i]),
            "non_pruned_index_out": dict_to_tensor(non_pruned_index_out[i]),
        }

        torch.save(checkpoint, save_path)

        print(f"Pruning Metadata stored for Level-{args.pruning_steps + 1 - i}")

    print("Iterative Pruning complete")
    torch.save(model, "./saves/pruning_metadata/core_model.pt")
    exit()

    model_info = get_vit_info(non_pruned_weights=model.state_dict(), num_heads=orig_copy.config.num_attention_heads, core_model=True)
    print("Model Info:", model_info)
  
    for id, m in enumerate(model.modules()):
        if isinstance(m, ElasticViTSelfAttention):
            print("num_heads:", m.num_attention_heads, 'head_dims:', m.attention_head_size, 'all_head_size:', m.all_head_size, '=>')
            m.num_attention_heads = model_info["num_heads"]
            m.attention_head_size = m.query.out_features // m.num_attention_heads
            m.all_head_size = m.query.out_features
            print("num_heads:", m.num_attention_heads, 'head_dims:', m.attention_head_size, 'all_head_size:', m.all_head_size)
            print()

    if args.stochastic_depth:
        inject_stochastic_depth(model)

    # Fine-Tune the Core model
    print("================FINE-TUNING CORE MODEL/DESCENDANT MODEL LEVEL-1======================")
    fine_tuner_core(args, device, model, train_loader, val_loader)

    """
    NOTE: Update the Core model weights after fine-tuning : 
    We update the non-pruned weights after pruning Level-2 model (Which is the core model)
    """
    updated_core_weights = extract_vit_core_weights(model)
    # checkpoint = torch.load(f"/var/scratch/skalra/elastoformer_saves/pruning_metadata/{args.exp_name}_pruning_metadata_Level_2.pth")
    # checkpoint["non_pruned_weights"] = updated_core_weights
    # checkpoint["classifier"] = [model.classifier.weight.detach().clone(), model.classifier.bias.detach().clone()]
    # torch.save(checkpoint, f"/var/scratch/skalra/elastoformer_saves/pruning_metadata/{args.exp_name}_pruning_metadata_Level_2.pth")
    non_pruned_weights_recorder["Level_2"] = updated_core_weights
    print("Updated Pruning Metadata for Level-2")
    
    if args.test_accuracy:
        print("Testing accuracy of the pruned model...")
        acc_pruned, loss_pruned = evaluate(model, criterion, val_loader, device=device, dist=args.distributed)
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

            rebuilding_weights = non_pruned_weights_recorder[f"Level_{i+2}"]
            pruned_weights = pruned_weights_recorder[f"Level_{i+2}"]
            rebuild_dim = get_vit_info(pruned_weights, rebuilding_weights, num_heads=(6 if "small" in args.model_name else None))
            print("Model Info:", rebuild_dim)
            rebuilt_model = create_vit_general(dim_dict=rebuild_dim, num_classes=num_classes).to(device)
            rebuilt_model,_,non_pruned_index_mapped = update_vit_weights_global(rebuilt_model, [pruned_index_in[args.pruning_steps-i-1], pruned_index_out[args.pruning_steps-i-1]], 
                                               [non_pruned_index_in[args.pruning_steps-i-1], non_pruned_index_out[args.pruning_steps-i-1]], 
                                               pruned_weights_recorder[f"Level_{i+2}"], non_pruned_weights_recorder[f"Level_{i+2}"], device=device)
            
            # partial freezing of grads for freezing the core weights
            freeze_partial_weights(rebuilt_model, non_pruned_index_mapped[0], non_pruned_index_mapped[1], device)
            
            if args.stochastic_depth:
                inject_stochastic_depth(rebuilt_model)
            fine_tuner(args, device, rebuilt_model, train_loader, val_loader, rebuild=True, in_freeze_indices=non_pruned_index_mapped[0], out_freeze_indices=non_pruned_index_mapped[1])

            if i < args.pruning_steps - 1:
                updated_weights = extract_vit_core_weights(rebuilt_model)
                # checkpoint = torch.load(f"/var/scratch/skalra/elastoformer_saves/pruning_metadata/{args.exp_name}_pruning_metadata_Level_{i+3}.pth")
                # checkpoint["non_pruned_weights"] = updated_weights
                # checkpoint["classifier"] = [rebuilt_model.classifier.weight.detach().clone(), rebuilt_model.classifier.bias.detach().clone()]
                # torch.save(checkpoint, f"/var/scratch/skalra/elastoformer_saves/pruning_metadata/{args.exp_name}_pruning_metadata_Level_{i+3}.pth")
                non_pruned_weights_recorder[f"Level_{i+3}"] = updated_weights
                print(f"Updated Pruning Metadata for Level-{i+3}")
                
                # DELETE UNUSED TENSORS TO FREE MEMORY
                del updated_weights

            if args.test_accuracy:
                print("Testing accuracy of the rebuild model...")
                acc_rebuilt, loss_rebuilt = evaluate(rebuilt_model, criterion, val_loader, device=device, dist=args.distributed)
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
            del pruned_weights, rebuilding_weights, rebuilt_model
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
