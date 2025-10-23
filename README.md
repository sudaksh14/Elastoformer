# Elastoformer
Here we provide the code for Elastoformer: Enabling Dynamic Adaptivity via Elastic Model Transformation

## Environment
Install all dependencies in requirements.txt before cloning this repo

## Usage
Here is the command for running the elastic transformation for DeiT:
```
python main.py \
    --exp_name elastoformer \
    --dataset_name imagenet \
    --model_name facebook/deit-base-patch16-224 \
    --pruning_type l1 \
    --pruning_ratio 0.5 \
    --iterative --pruning_steps 5 \
    --data_path /path/to/dataset \
    --train_batch_size 512 \
    --val_batch_size 512 \
    --save_as /path/for/save/ \
    --test_accuracy \ 
    --rebuild \
    --epochs 50 --core_epochs 50 --lr-warmup-epochs 10 --lr 5e-5 \
    --mixup-alpha 0.2 --core_weight_decay 0.05 --stochastic_depth \
    --clip-grad-norm 1 --amp --ra-sampler \
    --distributed --model-ema --log_wandb \
```
where:
- `--dataset_name`: [`imagenet`, `cifar10`, `cifar100`, `dummy`].
- `--model_name`: network architecture, choices [`facebook/deit-base-patch16-224`, `facebook/deit-small-patch16-224`, `google/vit_large_patch16_224`].
- `--pruning_type`: [`l1`, `l2`, `taylor`, `hessian`].
- `--pruning_ratio`: `Between (0,1) for the level of compression, 1 being max compression`.
- `--iterative`: `Flag for multiple step of elasticity (always True)`.
- `--pruning_steps`: `Desired number of Descendant Networks (DNs), Note: # DN's = pruning_steps + 1`.
- `--rebuild`: `Flag for retraining of DNs (always True)`.
