import logging
import os
import sys
import random
from datetime import datetime

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import create_model_and_transforms, trace_model, get_tokenizer
from open_clip.factory import image_transform
from training.data import get_data
from training.distributed import is_master, init_distributed_device, world_info_from_env
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr
from training.train import train_one_epoch, evaluate, unwrap_model
from training.train import evaluate_cinemanet
from training.data import get_imagenet, DataInfo

from upyog.all import *
from upyog.cli import Param as P
from types import SimpleNamespace


if torch.cuda.is_available():
    # This enables tf32 on Ampere GPUs which is only 8% slower than
    # float16 and almost as accurate as float32
    # This was a default in pytorch until 1.12
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


if True:
    import sys
    sys.path.append("/home/synopsis/git/CinemaNet-Training/")
    sys.path.append("/home/synopsis/git/YOLOX-Custom/")
    sys.path.append("/home/synopsis/git/YOLO-CinemaNet/")
    sys.path.append("/home/synopsis/git/icevision/")
    sys.path.append("/home/synopsis/git/labelling-workflows/")
    sys.path.append("/home/synopsis/git/amalgam/")
    sys.path.append("/home/synopsis/git/cinemanet-multitask-classification/")
    sys.path.append("/home/synopsis/git/Synopsis.py/")

from cinemanet.CLIP.inference import run_image_classification
from cinemanet.CLIP.utils import interpolate_weights
from cinemanet.CLIP.mapping import TAXONOMY
from cinemanet.CLIP.utils import load_model

def evaluate_model(
    # Main Args
    variant:       P("Model arch", str) = "ViT-L-14",
    pretrained:    P("Pretrained?", str) = "openai",
    ckpt_path:     P("", str) = None,
    alphas:        P("Alpha to blend `pretrained` and `ckpt_path` model", float, nargs='+') = None,
    experiment:    P("Name of the experiment", str) = None,
    # ...
    wandb_id:      P("Wandb ID to log runs to", str) = None,
    imagenet:      P("Do ImageNet val?", bool) = True,
    cinemanet:     P("Do CinemaNet val?", bool) = True,
    batch_size:    P("Batch Size", int) = 64,
    num_workers:   P("DataLoader num workers", int) = 4,
    device:        P("Device", int) = 0,
    path_imagenet: P("Path to the ImageNet validation set", str) = "/home/synopsis/datasets/ImageNet/validation/",
    save_dir:      P("Path to log metrics to", str) = None,
) -> pd.DataFrame:
    if pretrained and ckpt_path:
        if alphas is None:
            raise RuntimeError(
                f"You provided a pretrained model & checkpoint path to blend, but did not "
                f"provide alpha values to do the blending"
            )
    if save_dir is None:
        save_dir = Path.cwd() / "val_logs"
        save_dir.mkdir(exist_ok=True)
        logger.warning(
            f"You did not input a save path, so the current directory is being used: "
            f""
        )
    args = SimpleNamespace(
        # Model
        model = variant,
        # Dataset
        batch_size = batch_size,
        workers = num_workers,
        imagenet_val = path_imagenet,
        # Eval
        zeroshot_frequency = 1,
        distributed = False,
        wandb = wandb_id is not None,
        val_frequency = 1,
        precision = "amp_bf16",
        device = device,
        save_logs = False,
        rank = 0,
    )

    tokenizer = get_tokenizer(args.model)
    model = load_model(variant, device, None, ckpt_path)
    model.device = torch.device(device)
    model.eval()

    sd_stock = torch.load("/home/synopsis/git/open_clip/src/openai-ViT-L-14_pretrained.pth", map_location="cpu")
    sd_alpha_10 = {k:v.detach().cpu() for k,v in unwrap_model(model).state_dict().items()}

    alphas = alphas or []
    pbar = tqdm(sorted(alphas))

    IMAGENET_METRICS = {}
    CINEMANET_METRICS = {}

    for alpha in pbar:
        name = f"alpha-{alpha}"
        pbar.set_description(name)

        weights = interpolate_weights(sd_alpha_10, sd_stock, alpha=alpha)
        unwrap_model(model).load_state_dict(weights)

        if imagenet:
            imagenet_data = {}
            image_mean = getattr(model.visual, 'image_mean', None)
            image_std = getattr(model.visual, 'image_std', None)
            preproc = image_transform(
                model.visual.image_size, False, image_mean, image_std)
            imagenet_data["imagenet-val"] = get_imagenet(args, (None, preproc), split="val")

            IMAGENET_METRICS[alpha] = evaluate(model, imagenet_data, 1, args)
            print(f"ImageNet Metrics (Alpha={alpha})")
            print(IMAGENET_METRICS[alpha])

        if cinemanet:
            accuracies = {}
            for category in TAXONOMY.keys():
                acc,_,_,_,_ = run_image_classification(model, tokenizer, category, batch_size=8, verbose=False)
                accuracies[category] = acc
            CINEMANET_METRICS[alpha] = accuracies
            print(f"CinemaNet Metrics (Alpha={alpha})")
            print(CINEMANET_METRICS[alpha])

        pbar.update()

    dfs = []
    for (alpha,cnet_data), (_,imgnet_data) in zip(CINEMANET_METRICS.items(), IMAGENET_METRICS.items()):
        data = {}
        data["ImgNet-Top-1"] = imgnet_data["imagenet-zeroshot-val-top1"]
        data["ImgNet-Top-5"] = imgnet_data["imagenet-zeroshot-val-top5"]
        for k,v in cnet_data.items():
            data[k] = v

        x = pd.DataFrame(data.items())
        x.columns = ["Categories", "Accuracy"]
        x.insert(0, 'Alpha', [alpha]*len(x))
        dfs.append(x)
    df = pd.concat(dfs)
    df.insert(0, 'Variant', [variant]*len(df))
    df.insert(0, 'Experiment', [experiment]*len(df))
    df.to_csv(save_dir / f"{experiment}_metrics", index=False)

    return df
