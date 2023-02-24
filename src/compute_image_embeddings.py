import sys
import torch

from upyog.all import *
from upyog.cli import Param as P
from cinemanet.CLIP.inference import EVALUATION_PROMPTS

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

from training.inference import InferenceModelFromDisk


@call_parse
def run_embeddings(
    arch:          P("Model arch", str) = "ViT-L-14",
    pretrained:    P("Pretrained?", str) = "openai",
    ckpt_path:     P("Path to the checkpoint. If `None`, the stock model is used", str) = None,
    img_folders:   P("(Optional) Folders with images to analyse", str, nargs="+") = ["/home/synopsis/datasets/shotdeck-thumbs/"],
    img_files:     P("(Optional) List of image files to analyses", str, nargs="+") = None,
    img_files_json: P("(Optional) A JSON file that contains a list of filepaths to analyse") = "/home/synopsis/git/CinemaNet-Training/assets/shotdeck_sample_830k.json",
    alpha:         P("If using `ckpt_path`, alpha value to blend the pretrained & finetuned model", float) = None,
    exp_name:      P("Name of the experiment. Acts as suffix to add to .feather filename", str) = "shotdeck_sample_embeddings",
    batch_size:    P("Batch Size", int) = 32,
    num_workers:   P("DataLoader num workers", int) = 4,
    device:        P("Device", int) = 0,
    save_dir:      P("Path to save the DataFrame to", str) = "/home/synopsis/datasets/serialised-datasets/CLIP/",
) -> Tuple[InferenceModelFromDisk, pd.DataFrame]:

    args = deepcopy(locals())
    rich.print(f"\nArgs:\n{args}\n")

    inf = InferenceModelFromDisk(
                   arch = arch,
                 device = device,
                  alpha = alpha,
             pretrained = pretrained,
              ckpt_path = ckpt_path,
         path_embedding = None,
        experiment_name = exp_name,
    )
    rich.print(f"Model Info:\n{inf}\n")

    if img_files_json is not None:
        if img_folders or img_files:
            logger.warning(
                f"Ignoring `img_files` and `img_folders` and using `img_files_json` to select files to be analysed"
            )
            img_folders = None
            img_files = load_json(img_files_json)

    df = inf.get_image_embeddings(img_files, img_folders, batch_size, num_workers, save_dir)

    save_path_json = str(inf.path_embedding).replace(".feather", ".json")
    with open(save_path_json, "w") as f:
        json.dump(args, f, indent=4)
    rich.print(f"Wrote accompanying metadata as JSON to {save_path_json}")

    return inf, df
