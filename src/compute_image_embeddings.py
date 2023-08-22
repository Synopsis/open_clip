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
    sys.path.append("/home/synopsis/git/CinemaNet-CLIP/")
    sys.path.append("/home/synopsis/git/YOLOX-Custom/")
    sys.path.append("/home/synopsis/git/YOLO-CinemaNet/")
    sys.path.append("/home/synopsis/git/icevision/")
    sys.path.append("/home/synopsis/git/labelling-workflows/")
    sys.path.append("/home/synopsis/git/amalgam/")
    sys.path.append("/home/synopsis/git/cinemanet-multitask-classification/")
    sys.path.append("/home/synopsis/git/Synopsis.py/")

from training.inference import InferenceModel


@call_parse
def run_embeddings(
    arch:           P("(Optional) Model arch", str) = None,
    pretrained:     P("(Optional) Pretrained?", str) = None,
    ckpt_path:      P("(Optional) Path to the checkpoint. If `None`, a pretrained model is used", str) = None,
    img_files_json: P("(Optional) A JSON file that is a list of filepaths to run inference on. `/home/synopsis/git/CinemaNet-Training/assets/*sample*json` has a bunch of these ready to go") = "/home/synopsis/git/CinemaNet-Training/assets/shotdeck_sample_830k.json",
    alphas:         P("(Optional) If using `ckpt_path`, alpha values to blend the pretrained & finetuned model", float, nargs='+') = [1.0],
    batch_size:     P("Batch Size", int) = 32,
    num_workers:    P("DataLoader num workers", int) = 4,
    device:         P("Device", int) = 0,
    save_dir:       P("(Optional) Path to save the DataFrame to. If using a trained ckpt, cached embeddings are saved in the root folder", str) = "/home/synopsis/datasets/serialised-datasets/CLIP/CLIP-Embeddings-Cached/",
    img_size:       P("(Optional) Img size for inference. Model's default size is used if not specified.", int) = None,
) -> Tuple[InferenceModel, pd.DataFrame]:
    """
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
    """

    args = deepcopy(locals())
    rich.print(f"\nArgs:\n{args}\n")

    img_files_json = Path(img_files_json)
    img_files = load_json(img_files_json)

    if ckpt_path and alphas:
        USING_TRAINED_CKPT = True
        if save_dir:
            logger.warning(
                f"`save_dir` will be ignored as we're using a trained checkpoint. The embeddings will be saved inside the checkpoint's root folder"
            )
        save_dir = Path(ckpt_path).parent.parent / f"prompt-matches__{img_files_json.stem}"
        experiment_suffix = Path(ckpt_path).parent.parent.name[:19]  # Extracts unique timestamp

    else:
        USING_TRAINED_CKPT = False
        if alphas:
            logger.warning(f"Ignoring `alphas` as a pretrained model is being used")
        alphas = [0.0]
        assert save_dir, f"`save_dir` needs to be passed if using a pretrained model"
        experiment_suffix = f"pretrained__{img_files_json.stem}"

    pbar = tqdm(alphas)
    for alpha in pbar:
        if USING_TRAINED_CKPT:
            pbar.set_description(f"EVALUATING AT ALPHA = {alpha}")
            inf = InferenceModel.from_ckpt_path(
                 ckpt_path = ckpt_path,
                    device = device,
                     alpha = alpha,
                batch_size = batch_size,
            )
        else:
            pbar.set_description(f"EVALUATING W/ Pre-Trained Model")
            inf = InferenceModel(
                           arch = arch,
                     pretrained = pretrained,
                         device = device,
                      ckpt_path = ckpt_path,
            )

        df = inf.get_image_embeddings(img_files, None, batch_size, num_workers, img_size)
        cached_embeddings_path = inf.save_embedding_df(df, experiment_suffix, save_dir)

        if not USING_TRAINED_CKPT:
            save_path_json = cached_embeddings_path.with_suffix(".json")
            with open(save_path_json, "w") as f:
                json.dump(args, f, indent=4)
            rich.print(f"Wrote accompanying metadata as JSON to {save_path_json}")

    return inf, df
