```bash
# BS
git config --global --add safe.directory /home/synopsis/git/open_clip
cp /home/synopsis/.netrc ~/

# ConvNeXT architectures
export VARIANT="convnext_base_w"  # 256x256

# convnext_base_w pretrained tags
# We don't use `augreg` because it uses too much augmentation which makes the network
# feature invariant to aspects that are cinematographically relevant.
# export PRETRAINED="laion2b_s13b_b82k_augreg"  # ImgNet 71.5%
export PRETRAINED="laion_aesthetic_s13b_b82k" # ImgNet 71.0%, aesthetic 900m subset only

# Freezing Vision Encoder
# See https://github.com/Synopsis/CinemaNet-CLIP/blob/main/notebooks/arch%20--%20convnext%20breakdown.ipynb for more details
export NUM_TUNABLE_LAYERS_VISION=5  # Final head + Last Conv Stage
# export NUM_TUNABLE_LAYERS_VISION=11  # Final head + Last Conv Stage + Some blocks. This has worked well in our experiments


# Freezing Text Encoder
# There are 12 Blocks in the Text Transformer
# export NUM_TUNABLE_LAYERS_TEXT=2  # Final head & LayerNorm
export NUM_TUNABLE_LAYERS_TEXT=3  # Final head & LayerNorm + Last Block
# export NUM_TUNABLE_LAYERS_TEXT=4  # Final head & LayerNorm + Last 2 Blocks


# Setup Dataset
export DSET_TYPE="cinema_single_caption"
# export DSET_TYPE="cinema_multi_caption"
# export DSET_TYPE="cinema_dynamic_caption"

export TRAIN_DATA_PATH="/home/synopsis/datasets/serialised-datasets/CLIP/CLIP-Training/Fine-Tuning-Playbook-Experiments/shot_angle__10k.feather"
export CAPTION_KEY="caption"
# export CAPTION_KEY="captions"


# Other HParams
export LR=1e-4             # Worth experimenting with
export EPOCHS=5            # Worth experimenting with
export BATCH_SIZE=256      # Make this as high as possible. Max value depends on other params
export ACCUM_FREQ=8        # Make this as high as possible. Max value depends on other params
export WARMUP_STEPS=100    # Adjust depending on total no. of steps. You want this to be a fraction of num_steps_total

# FORMULA FOR STEPS (Round up for exact number):
# steps_per_epoch = NUM_TRAINING_SAMPLES / (BATCH_SIZE*ACCUM_FREQ))
# total_steps = EPOCHS * steps_per_epoch


# ---- LAUNCH TRAINING RUN ---- #

# If you want to run on a single CPU, launch with `python training/main.py`
torchrun --nproc_per_node 3 -m training.main \
    --train-data $TRAIN_DATA_PATH \
    --dataset-type $DSET_TYPE \
    --csv-img-key filepath_mistake_not \
    --csv-caption-key $CAPTION_KEY \
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
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --accum-freq $ACCUM_FREQ \
    --lock-image \
    --lock-image-unlocked-groups $NUM_TUNABLE_LAYERS_VISION \
    --lock-text \
    --lock-text-unlocked-layers $NUM_TUNABLE_LAYERS_TEXT
```
