import datetime
import os
import time
import warnings
import wandb
import torch
import torch.utils.data
import utils.train_utils as train_utils
from torch import nn
from utils.prune_utils import selective_gradient_clipping_norm
from utils.cnn_prune_utils import selective_gradient_clipping_norm_cnn
from datasets import mixup_fn


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None, CNN=False, mixup_fn=None):
    model.train()
    metric_logger = train_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", train_utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", train_utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        if mixup_fn is not None:
            image, target = mixup_fn(image, target)
        image, target = image.to(device), target.to(device)
        with torch.autocast(device_type="cuda", enabled=scaler is not None):
            if CNN:
                output = model(image)
            else:
                output = model(image).logits
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = train_utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

    return metric_logger

def train_one_epoch_freeze(model, in_freeze_indices, out_freeze_indices, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None, CNN=False, mixup_fn=None):
    model.train()
    metric_logger = train_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", train_utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", train_utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        if mixup_fn is not None:
            image, target = mixup_fn(image, target)
        image, target = image.to(device), target.to(device)
        with torch.autocast(device_type="cuda", enabled=scaler is not None):
            if CNN:
                output = model(image)
            else:
                output = model(image).logits
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                if args.distributed:
                    selective_gradient_clipping_norm(model.module, out_freeze_indices, in_freeze_indices, args.clip_grad_norm, device)
                else:
                    selective_gradient_clipping_norm(model, out_freeze_indices, in_freeze_indices, args.clip_grad_norm, device)
            
            scaler.step(optimizer)
            scaler.update()
        
        else:
            loss.backward()
            
            if args.clip_grad_norm is not None:
                if args.distributed:
                    selective_gradient_clipping_norm_cnn(model.module, out_freeze_indices, in_freeze_indices, args.clip_grad_norm, device)
                else:
                    selective_gradient_clipping_norm_cnn(model, out_freeze_indices, in_freeze_indices, args.clip_grad_norm, device)

            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = train_utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

    return metric_logger


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix="", dist=False, CNN=False):
    model.eval()
    metric_logger = train_utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if CNN:
                output = model(image)
            else:
                output = model(image).logits
            loss = criterion(output, target)

            acc1, acc5 = train_utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    if dist:
        num_processed_samples = train_utils.reduce_across_processes(num_processed_samples)
        if (
            hasattr(data_loader.dataset, "__len__")
            and len(data_loader.dataset) != num_processed_samples
            and torch.distributed.get_rank() == 0
        ):
            # See FIXME above
            warnings.warn(
                f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
                "samples were used for the validation, which might bias the results. "
                "Try adjusting the batch size and / or the world size. "
                "Setting the world size to 1 is always a safe bet."
            )

        metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg, metric_logger.loss.global_avg


def fine_tuner(args, device, model, data_loader, data_loader_test, rebuild=False, in_freeze_indices=None, out_freeze_indices=None):
    if args.output_dir:
        train_utils.mkdir(args.output_dir)

    # if args.distributed:
    #     utils.init_distributed_mode(args)
    # print(args)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = train_utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop, Adam and AdamW are supported.")

    scaler = torch.amp.GradScaler('cuda') if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.train_batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = train_utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA", dist=args.distributed, CNN=(args.model_name.startswith("resnet") or args.model_name.startswith("vgg")))
        else:
            evaluate(model, criterion, data_loader_test, device=device, dist=args.distributed, CNN=(args.model_name.startswith("resnet") or args.model_name.startswith("vgg")))
        return

    if args.log_wandb:
        if args.distributed:
            if train_utils.get_rank() == 0:
                wandb.init(
                    project="ViT-ImageNet-Adaptivity",
                    name=f"{args.exp_name}_Pruning_{args.pruning_type}_{args.pruning_ratio}",
                    config=vars(args)  # <--- Converts argparse.Namespace to dict!
                )
        else:
            wandb.init(
                project="ViT-ImageNet-Adaptivity",
                name=f"{args.exp_name}_Pruning_{args.pruning_type}_{args.pruning_ratio}",
                config=vars(args)  # <--- Converts argparse.Namespace to dict!
            )

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if rebuild:
            metrics = train_one_epoch_freeze(model, in_freeze_indices, out_freeze_indices, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler, 
                                             CNN=(args.model_name.startswith("resnet") or args.model_name.startswith("vgg")), mixup_fn=mixup_fn)
        else:
            metrics = train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler, 
                                      CNN=(args.model_name.startswith("resnet") or args.model_name.startswith("vgg")), mixup_fn=mixup_fn)
        
        test_acc1,_ = evaluate(model, criterion, data_loader_test, device=device, dist=args.distributed, CNN=(args.model_name.startswith("resnet") or args.model_name.startswith("vgg")))
        if model_ema:
            test_ema_acc1,_ = evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA", dist=args.distributed, CNN=(args.model_name.startswith("resnet") or args.model_name.startswith("vgg")))
        
        wandb_metrics = metrics.get_all_averages()

        if args.log_wandb:
            if args.distributed:
                if train_utils.get_rank() == 0:
                    wandb.log({"epoch": epoch,
                                **wandb_metrics,  # Spread all logged metrics
                                "test_acc1": test_acc1,
                                "test_ema_acc1": test_ema_acc1,
                            })
                    
            else:
                wandb.log({"epoch": epoch,
                            **wandb_metrics,  # Spread all logged metrics
                            "test_acc1": test_acc1,
                            "test_ema_acc1": test_ema_acc1,
                        })

        lr_scheduler.step()

        # if args.output_dir:
        #     checkpoint = {
        #         "model": model_without_ddp.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #         "lr_scheduler": lr_scheduler.state_dict(),
        #         "epoch": epoch,
        #         "args": args,
        #     }
        #     if model_ema:
        #         checkpoint["model_ema"] = model_ema.state_dict()
        #     if scaler:
        #         checkpoint["scaler"] = scaler.state_dict()
        #     utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
        #     utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def fine_tuner_core(args, device, model, data_loader, data_loader_test):
    if args.output_dir:
        train_utils.mkdir(args.output_dir)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = train_utils.set_weight_decay(
        model,
        args.core_weight_decay,
        norm_weight_decay=args.core_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.core_weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.core_weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.core_weight_decay)
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.core_weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop, Adam and AdamW are supported.")

    scaler = torch.amp.GradScaler('cuda') if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.core_epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.train_batch_size * args.model_ema_steps / args.core_epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = train_utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA", dist=args.distributed)
        else:
            evaluate(model, criterion, data_loader_test, device=device, dist=args.distributed)
        return

    if args.log_wandb:
        if args.distributed:
            if train_utils.get_rank() == 0:
                wandb.init(
                    project="ViT-ImageNet-Adaptivity",
                    name=f"{args.exp_name}_Pruning_{args.pruning_type}_{args.pruning_ratio}",
                    config=vars(args)  # <--- Converts argparse.Namespace to dict!
                )
        else:
            wandb.init(
                project="ViT-ImageNet-Adaptivity",
                name=f"{args.exp_name}_Pruning_{args.pruning_type}_{args.pruning_ratio}",
                config=vars(args)  # <--- Converts argparse.Namespace to dict!
            )

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.core_epochs):

        metrics = train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler, 
                                  CNN=(args.model_name.startswith("resnet") or args.model_name.startswith("vgg")), mixup_fn=mixup_fn)
        test_acc1,_ = evaluate(model, criterion, data_loader_test, device=device, dist=args.distributed, CNN=(args.model_name.startswith("resnet") or args.model_name.startswith("vgg")))
        if model_ema:
            test_ema_acc1,_ = evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA", dist=args.distributed, CNN=(args.model_name.startswith("resnet") or args.model_name.startswith("vgg")))
        
        wandb_metrics = metrics.get_all_averages()

        if args.log_wandb:
            if args.distributed:
                if train_utils.get_rank() == 0:
                    wandb.log({"epoch": epoch,
                                **wandb_metrics,  # Spread all logged metrics
                                "test_acc1": test_acc1,
                                "test_ema_acc1": test_ema_acc1,
                            })
                    
            else:
                wandb.log({"epoch": epoch,
                            **wandb_metrics,  # Spread all logged metrics
                            "test_acc1": test_acc1,
                            "test_ema_acc1": test_ema_acc1,
                        })

        lr_scheduler.step()

        # if args.output_dir:
        #     checkpoint = {
        #         "model": model_without_ddp.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #         "lr_scheduler": lr_scheduler.state_dict(),
        #         "epoch": epoch,
        #         "args": args,
        #     }
        #     if model_ema:
        #         checkpoint["model_ema"] = model_ema.state_dict()
        #     if scaler:
        #         checkpoint["scaler"] = scaler.state_dict()
        #     utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
        #     utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")