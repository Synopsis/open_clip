import sys
import torch

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
from cinemanet.CLIP.utils import load_model, load_interpolated_model


@call_parse
def run_embeddings(
    variant:       P("Model arch", str) = "ViT-L-14",
    pretrained:    P("Pretrained?", str) = "openai",
    ckpt_path:     P("Path to the checkpoint. If `None`, the stock model is used", str) = None,
    img_folders:   P("(Optional) Folders with images to analyse", str, nargs="+") = ["/home/synopsis/datasets/shotdeck-thumbs/"],
    img_files:     P("(Optional) List of image files to analyses", str, nargs="+") = None,
    img_files_json: P("(Optional) A JSON file that contains a list of filepaths to analyse") = "/home/synopsis/git/CinemaNet-Training/assets/shotdeck_sample_100k.json",
    alpha:         P("If using `ckpt_path`, alpha value to blend the pretrained & finetuned model", float) = None,
    suffix:        P("Suffix to add to .feather filename", str) = "shotdeck_sample_embeddings",
    batch_size:    P("Batch Size", int) = 32,
    num_workers:   P("DataLoader num workers", int) = 4,
    device:        P("Device", int) = 0,
    save_dir:      P("Path to save the DataFrame to", str) = "/home/synopsis/datasets/serialised-datasets/CLIP/",
) -> Tuple[pd.DataFrame, dict]:
    args = deepcopy(locals())
    rich.print(f"\nArgs:\n{args}\n")

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
    filename = f"{variant}--{pretrained}"
    if ckpt_path:
        filename = f"{filename}--finetuned-alpha-{str(alpha)}"
    filename = f"{filename}__{suffix}"
    save_path_df = save_dir / f"{filename}.feather"
    save_path_json = save_dir / f"{filename}_args.json"

    if ckpt_path is not None:
        model = load_interpolated_model(variant, device, pretrained, ckpt_path, alpha=alpha)
        rich.print(f"Loaded interpolated weights at Alpha={alpha}")
    else:
        model = load_model(variant, device, pretrained, ckpt_path)
        rich.print(f"Loaded pretrained model")

    # model = load_model(variant, device, None, ckpt_path)
    # model.eval()

    # sd_stock = torch.load("/home/synopsis/git/open_clip/src/openai-ViT-L-14_pretrained.pth", map_location="cpu")
    # sd_finetune = {k:v.detach().cpu() for k,v in model.state_dict().items()}
    # interpolated_wts = interpolate_weights(sd_finetune, sd_stock, alpha)

    # model.load_state_dict(interpolated_wts)
    # rich.print(f"Loaded interpolated weights at Alpha={alpha}")

    if img_files_json is not None:
        if img_folders or img_files:
            logger.warning(
                f"Ignoring `img_files` and `img_folders` and using `img_files_json` to select files to be analysed"
            )
            img_folders = None
            img_files = load_json(img_files_json)

    df = compute_image_embeddings(model, img_files, img_folders, batch_size, num_workers)
    df.to_feather(save_path_df)

    with open(save_path_json, "w") as f:
        json.dump(args, f, indent=4)
    rich.print(f"Wrote cached embeddings as DataFrame to {save_path_df}")
    rich.print(f"Wrote accompanying metadata as JSON to {save_path_json}")

    return df, args
