import sys
import torch
from datetime import datetime

from open_clip import get_tokenizer
from open_clip.factory import image_transform
from training.train import evaluate, unwrap_model
from training.data import get_imagenet

from upyog.all import *
from upyog.cli import Param as P


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

from cinemanet.CLIP.inference import compute_image_embeddings
from cinemanet.CLIP.utils import interpolate_weights
from cinemanet.CLIP.utils import load_model


@call_parse
def run_embeddings(
    variant:       P("Model arch", str) = "ViT-L-14",
    pretrained:    P("Pretrained?", str) = "openai",
    ckpt_path:     P("", str) = None,
    img_folders:   P("", str, nargs="+") = ["/home/synopsis/datasets/shotdeck-thumbs/"],
    img_files:     P("", str, nargs="+") = None,
    alpha:         P("Alpha to blend `pretrained` and `ckpt_path` model", float) = None,
    experiment:    P("Name of the experiment", str) = None,
    batch_size:    P("Batch Size", int) = 64,
    num_workers:   P("DataLoader num workers", int) = 4,
    device:        P("Device", int) = 0,
    save_dir:      P("Path to save the DataFrame to", str) = "/home/synopsis/datasets/serialised-datasets/CLIP/",
) -> Tuple[pd.DataFrame, dict]:
    args = deepcopy(locals())

    if pretrained and ckpt_path:
        if alpha is None:
            raise RuntimeError(
                f"You provided a pretrained model & checkpoint path to blend, but did not "
                f"provide alpha values to do the blending"
            )

    if not save_dir:
        raise RuntimeError(f"You need to enter a `save_dir`")

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    save_path_df = save_dir / f"{experiment}.feather"
    save_path_json = save_dir / f"{experiment}_args.json"

    model = load_model(variant, device, None, ckpt_path)
    model.eval()

    sd_stock = torch.load("/home/synopsis/git/open_clip/src/openai-ViT-L-14_pretrained.pth", map_location="cpu")
    sd_finetune = {k:v.detach().cpu() for k,v in unwrap_model(model).state_dict().items()}
    interpolated_wts = interpolate_weights(sd_finetune, sd_stock, alpha)

    model.load_state_dict(interpolated_wts)
    rich.print(f"Loaded interpolated weights at Alpha={alpha}")

    df = compute_image_embeddings(model, img_files, img_folders, batch_size, num_workers)
    df.to_feather(save_path_df)

    with open(save_path_json, "w") as f:
        json.dump(args, f, indent=4)
    rich.print(f"Wrote cached embeddings as DataFrame to {save_path_df}")
    rich.print(f"Wrote accompanying metadata as JSON to {save_path_json}")

    return df, args
