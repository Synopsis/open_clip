```bash
git config --global --add safe.directory /home/synopsis/git/open_clip
# export VARIANT="ViT-L-14"
# export VARIANT="ViT-L-14-336"

# For using an OpenAI ckpt (only available with some archs, and not the convnext ones)
# export PRETRAINED="openai"


pip install timm --upgrade --no-deps  # We need >= 0.6.12
# ConvNeXT architectures
export VARIANT="convnext_base_w"  # 256x256

# convnext_base_w pretrained tags
export PRETRAINED="laion2b_s13b_b82k_augreg"  # ImgNet 71.5%
# export PRETRAINED="laion2b_s13b_b82k"         # ImgNet 70.8%
# export PRETRAINED="laion_aesthetic_s13b_b82k" # ImgNet 71.0%, aesthetic 900m subset only

# number of layers (starting from the last one) to be left unlocked
# convnext_base_w has 37 layers
#  4 = head + last conv stage
# 31 = head + last 2 conv stages (2nd last one is pretty fat.)
# export NUM_TUNABLE_LAYERS=4
export NUM_TUNABLE_LAYERS=11  # ~70% of the encoder remained frozen
# export NUM_TUNABLE_LAYERS=31

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
    --pretrained $PRETRAINED \
    --precision amp_bf16 \
    --lr=1e-4 \
    --warmup 500 \
    --model $VARIANT \
    --epochs 5 \
    --batch-size=256 \
    --accum-freq 8 \
    --lock-image \
    --lock-image-unlocked-groups $NUM_TUNABLE_LAYERS
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