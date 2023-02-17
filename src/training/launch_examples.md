```bash
git config --global --add safe.directory /home/synopsis/git/open_clip
# export VARIANT="ViT-L-14"
export VARIANT="ViT-L-14-336"
# export VARIANT="convnext_base_w"

torchrun --nproc_per_node 2 -m training.main \
    --train-data /home/synopsis/datasets/serialised-datasets/CLIP-Training/all-categories__multi-caption__thresh-0-9__644k-cinematic.csv \
    --dataset-type csv_multicaption \
    --csv-separator , \
    --csv-img-key filepath_mistake_not \
    --csv-caption-key captions \
    --wd=0.1 \
    --workers 8 \
    --report-to wandb,tensorboard \
    --local-loss \
    --use-bn-sync \
    --gather-with-grad \
    --grad-clip-norm 1.0 \
    --pretrained openai \
    --precision amp_bf16 \
    --lr=1e-4 \
    --warmup 500 \
    --model $VARIANT \
    --epochs 1 \
    --batch-size=64 \
    --accum-freq 8 \
    --lock-image \
    --lock-image-unlocked-groups 4
```


```bash
# 3 unfrozen; 0.9 thresh & multi-cap; CUSTOM TEXT ENCODER!!

git config --global --add safe.directory /home/synopsis/git/open_clip
VARIANT="ViT-L-14-custom-text"

torchrun --nproc_per_node 3 -m training.main \
    --train-data /home/synopsis/datasets/serialised-datasets/CLIP-Training/all-categories__custom-token-caption-verbose__thresh-0-9__644k-cinematic.csv \
    --dataset-type csv_multicaption \
    --csv-separator , \
    --csv-img-key filepath_mistake_not \
    --csv-caption-key captions \
    --wd=0.1 \
    --epochs 3 \
    --workers 8 \
    --report-to wandb,tensorboard \
    --local-loss \
    --use-bn-sync \
    --gather-with-grad \
    --grad-clip-norm 1.0 \
    --pretrained openai \
    --model $VARIANT \
    --custom-text-encoder \
    --precision amp_bf16 \
    --lr=1e-4 \
    --warmup 500 \
    --batch-size=112 \
    --accum-freq 3 \
    --lock-image \
    --lock-image-unlocked-groups 5
```