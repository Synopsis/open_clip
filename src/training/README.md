# Launching Training Runs

This section has sample commands to launch training runs.


## Environment Setup

First, we gotta set up an environment. For this, launch the docker container as follows:
```bash
cd /home/synopsis/git/Ozu-ML-Dev-Docker-Setups/open_clip/
./docker-run-mistake-not.sh
```

Once inside the container, `cd` into this subdirectory (where this README currently is) and run a few commands. These only need to be run once after the Docker container is initialiased:
```bash
cd git/open_clip/src/training/

# W&B Setup
git config --global --add safe.directory /home/synopsis/git/open_clip
cp /home/synopsis/.netrc ~/

# Install our fork locally
pip install -e ../../ --no-dependencies
```

---

## Args / HParams

In the training script below, we set a few bash variables for key hparams. Here's a more detailed description of them:
### Architecture Args
* `VARIANT`: Name of the the architecture. Must be one of `open_clip.list_models()`
* `PRETRAINED`: Name of the pretrained checkpoint to load from. See `open_clip.list_pretrained()` for options. We use the `'laion_aesthetic_s13b_b82k'` ckpt and not the `'augreg'` one (though it's slightly higher in accuracy) because the additional augmentation destroys features that are useful from a cinematic perspective

### Data Args
* `DSET_TYPE`: Should be any of the following options:
    - `'cinema_single_caption'` - A .feather file with a column for the filepath and another with a _single_ caption for the file
    - `'cinema_multi_caption'` - A .feather file with a column for the filepath and another with _multiple_ captions for the file
    - `'cinema_dynamic_caption'` - TODO

See [this notebook](https://github.com/Synopsis/CinemaNet-CLIP/blob/main/notebooks/data%20--%20debug%20training%20dataset%20classes.ipynb) to load some sample data
* `TRAIN_DATA_PATH`: Path to the `.feather` file i.e. the dataset. See `/home/synopsis/datasets/serialised-datasets/CLIP/CLIP-Training/Fine-Tuning-Playbook-Experiments` for a bunch of single-caption files
* `CAPTION_KEY`: Name of the column that contains the caption



### Model Freezing Args

These numbers are specific to the `convnext_base.*` architecture. More explorations in [this notebook](https://github.com/Synopsis/CinemaNet-CLIP/blob/main/notebooks/arch%20--%20convnext%20breakdown.ipynb) 
* `NUM_TUNABLE_LAYERS_VISION`: Number of vision encoder layers to be fine-tuned

| **NUM_TUNABLE_LAYERS_VISION** | **Components Being Tuned** |
| --- | --- |
| 2 | Head |
| 5 | Head + Last ConvNeXT Stage |
| 11 | Head + Last 2 ConvNeXT Stages |


* `NUM_TUNABLE_LAYERS_TEXT`: Number of text encoder layers to be fine-tuned

| **NUM_TUNABLE_LAYERS_TEXT** | **Components Being Tuned** |
| --- | --- |
| 2 | Head (w/ LayerNorm) |
| 3 | Head (w/ LayerNorm) + Last Block |
| 4 | Head (w/ LayerNorm) + Last 2 Blocks |
| 7 | Head (w/ LayerNorm) + Last 5 Blocks |
| 12 | Head (w/ LayerNorm) + Last 10 Blocks |
| 14 | Head (w/ LayerNorm) + ALL 12 Blocks |
| 16 | Head (w/ LayerNorm) + ALL 12 Blocks + Positional & Token Embedding |


### Tuning Args

* `LR`: Learning rate. Currently, a flat learning rate is set for all layers. One huge area for improvement is to use differential learning rates for later vs. earlier layers.
* `EPOCHS`: Num. training epochs
* `BATCH_SIZE`
* `ACCUM_FREQ`: Num. of batches to accumulate gradients for. `8` or `16` is the max (varies based on other args)
* `WARMUP_STEPS`: Num. of warmup steps. It's _crucial_ that you ensure this is well below the total number of training steps
* `EVAL_FREQUENCY`: How often to run evalutaion

Here is a reference formula for computing the No. of steps in total and per epoch:
```python
steps_per_epoch = NUM_TRAINING_SAMPLES / (BATCH_SIZE * ACCUM_FREQ * NUM_GPUS)
total_steps     = EPOCHS * steps_per_epoch
```

**Note on Inference:**
When training, we run inference on ImageNet, CinemaNet, and ShotDeck-CLIP validation sets. These are logged to W&B, along with confusion matrices (except for ImageNet).
At the end of training, we load up a 110k sample ShotDeck dataset and run a few "common" prompts and save their top 21 results to W&B.

For all of the above, inference is done at alphas `0.0`, `0.5`, `0.75` and `1.0`

To run eval on only select CinemaNet categories, use the `--cinemanet-eval-categories` arg (not set as a bash variable currently... derp)


## Launching The Training Script

```bash
# Arch
export VARIANT="convnext_base_w"
export PRETRAINED="laion_aesthetic_s13b_b82k"

# Data
export DSET_TYPE="cinema_single_caption"
export TRAIN_DATA_PATH="/home/synopsis/datasets/serialised-datasets/CLIP/CLIP-Training/Fine-Tuning-Playbook-Experiments/shot_angle__10k.feather"
export CAPTION_KEY="caption"

# Freezing
export NUM_TUNABLE_LAYERS_VISION=5
export NUM_TUNABLE_LAYERS_TEXT=7   # Final head & LayerNorm + Last 5 Blocks

# Tuning
export LR=1e-3
export EPOCHS=15
export BATCH_SIZE=256
export ACCUM_FREQ=8
export WARMUP_STEPS=3
export EVAL_FREQUENCY=5


# --------- LAUNCH TRAINING RUN --------- #
# If you want to run on a single CPU, launch with `python training/main.py`
# python training/main.py \
torchrun --nproc_per_node 3 -m main \
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
    --lock-text-unlocked-layers $NUM_TUNABLE_LAYERS_TEXT \
    --save-frequency $EVAL_FREQUENCY \
    --cinemanet-eval-categories shot_angle
```

---

## Inference (Outside Training)

The `inference.py` file has a class called `InferenceModel` that can run inference using either pretrained and/or trained checkpoints. See its docstring for more details on specific args.

<details><summary>
    Use the `../compute_image_embeddings.py` script to run inference and save cached embeddings using this class.
Pasted below is the output of `python ../compute_image_embeddings.py --help`:

</summary>

```
    This script lets you generate a cached embeddings `.feather` file with either
    pretrained or trained checkpoints of CLIP models

    If using a pretrained model, use these args: `arch`, `pretrained` and `save_dir`
    If using a trained checkpoint, use `ckpt_path` and `alphas`

    For trained checkpoints, the folder structure after the script will look something like this:
        <`ckpt_path`'s Grandparent Root Folder>
        ├── checkpoints
        │   ├── `ckpt_path`
        ├── out.log
        ├── params.txt
        ├── prompt-matches__<`img_files_json.stem`>
        │   ├── convnext_base_w--laion_aesthetic_s13b_b82k--finetuned-alpha-0.0__2023_02_26-13_34_33.feather
        │   ├── convnext_base_w--laion_aesthetic_s13b_b82k--finetuned-alpha-0.5__2023_02_26-13_34_33.feather
        └── tensorboard

        Where `prompt-matches__*` containes a `.feather` file saved for each value of `alphas`

    For pretrained checkpoints, the folder structure after the script will look something like this:
        <SAVE_DIR>
        ├── {arch}--{pretrained}__pretrained__{img_files_json.stem}.feather
        └── {arch}--{pretrained}__pretrained__{img_files_json.stem}.json
    

optional arguments:
  -h, --help         show this help message and exit
  --arch             (Optional) Model arch (default: ViT-L-14)
  --pretrained       (Optional) Pretrained? (default: openai)
  --ckpt_path        (Optional) Path to the checkpoint. If `None`, a pretrained model is used
  --img_files_json   (Optional) A JSON file that is a list of filepaths to run inference on.
                     `/home/synopsis/git/CinemaNet-Training/assets/*sample*json` has a bunch of these ready to go
                     (default: /home/synopsis/git/CinemaNet-Training/assets/shotdeck_sample_830k.json)
  --alphas  [ ...]   (Optional) If using `ckpt_path`, alpha values to blend the pretrained & finetuned model (default:
                     [0.0, 0.5, 0.75, 1.0])
  --batch_size       Batch Size (default: 32)
  --num_workers      DataLoader num workers (default: 4)
  --device           Device (default: 0)
  --save_dir         (Optional) Path to save the DataFrame to. If using a trained ckpt, cached embeddings are saved in the
                     root folder (default: /home/synopsis/datasets/serialised-datasets/CLIP/CLIP-Embeddings-Cached/)
```

</details>

With cached embeddings, you can use some of the tooling in the notebooks below

#### List of Relevant Notebooks

* [Evaluate on ImageNet, CinemaNet, ShotDeck-CLIP](https://github.com/rsomani95/open_clip/blob/no-center-crop/src/evaluate.ipynb)

These notebooks could use some cleaning up (and might currently be broken) as they've been messed around with quite a bit
* [Running inference with a single model](https://github.com/rsomani95/open_clip/blob/no-center-crop/src/inference-single-model.ipynb)
* [Interactive Environment To Test Multiple Models](https://github.com/rsomani95/open_clip/blob/no-center-crop/src/interactive-clip-explorer.ipynb)
    - Expects cached embeddings (see section above for how to compute these)
    - Path to trained models needs to be updated manually
