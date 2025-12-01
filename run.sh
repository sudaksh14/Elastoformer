# For Elastoformer (Vision Transformers)
python main.py \
    --exp_name elastoformer \
    --dataset_name imagenet \
    --model_name facebook/deit-base-patch16-224 \
    --pruning_type l1 \
    --pruning_ratio 0.5 \
    --iterative --pruning_steps 5 \
    --data_path /path/to/dataset/\
    --train_batch_size 512 \
    --val_batch_size 512 \
    --save_as /path/to/saves/ \
    --test_accuracy \
    --rebuild \
    --epochs 50 --core_epochs 50 --lr-warmup-epochs 10 --lr 5e-5 \
    --mixup-alpha 0.2 --core_weight_decay 0.05 --stochastic_depth \
    --clip-grad-norm 1 --amp --ra-sampler \
    --distributed --model-ema --log_wandb \

# For Elastic CNNs
python elastic_cnn.py \
    --exp_name Elastic_Resnet50 \
    --dataset_name imagenet \
    --model_name resnet50 \
    --weights ResNet50_Weights.IMAGENET1K_V2 \
    --pruning_type l1 \
    --pruning_ratio 0.5 \
    --iterative --pruning_steps 5 \
    --taylor_batchs 10 \
    --data_path /nvmestore/koelma/pytorch_work/ilsvrc2012/ \
    --train_batch_size 128 \
    --val_batch_size 128 \
    --save_as saves/state_dicts/ \
    --test_accuracy \
    --rebuild \
    --epochs 50 --core_epochs 50 --lr-warmup-epochs 5 --lr 0.05 \
    --mixup-alpha 0.8 --core_weight_decay 1e-4 --stochastic_depth \
    --clip-grad-norm 1 --amp --ra-sampler \
    --distributed --model-ema --log_wandb \

