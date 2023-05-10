from upyog.all import *
from open_clip import get_tokenizer
from open_clip.pretrained import get_pretrained_cfg
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

import torch

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


from cinemanet_clip.inference import (
    compute_image_embeddings, run_cinemanet_eval_by_category, run_shotdeck_clip_eval_by_category
)
from cinemanet_clip.utils.model_loading import (
    load_interpolated_model, load_model, interpolate_weights)
from cinemanet_clip.mapping import (
    TAXONOMY, TAXONOMY_CUSTOM_TOKENS, REVERSE_TAXONOMY, REVERSE_TAXONOMY_CUSTOM_TOKENS
)
from cinemanet_clip.inference.data_handler import DataHandlerCLIPValidation

DEFAULT_CINEMANET_CATEGORIES = [
    c for c in sorted(TAXONOMY.keys()) if c.startswith("color_") or c.startswith("shot_")
]

__all__ = ["ModelCfg", "InferenceModel", "InferenceModelWhileTraining"]


@dataclass
class ModelCfg:
    arch: str
    device: Union[str, int, torch.device]
    alpha: float
    pretrained: Optional[str]
    ckpt_path: Optional[str]
    # path_embedding: Optional[str]


def _load_params_txt(file_path):
    "Load the `params.txt` file saved by a training run"
    config_dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            key_value = line.split(": ", 1)
            key = key_value[0]
            value = key_value[1] if len(key_value) > 1 else ""
            config_dict[key] = value

    return config_dict


class InferenceModel:
    def __init__(
        self,
        arch: str,
        pretrained: str,
        device: Union[str, int, torch.device],
        ckpt_path: Optional[str] = None,
        alpha: Optional[float] = None,
        batch_size: int = 128,
    ):
        """
        Helper class to easily run inference for both pretrained and/or trained checkpoints.
        If loading a trained checkpoint, you can dynamically reset the `alpha` to interpolate weights on the fly

        ARGS:
            arch:       Name of the the architecture. Must be one of `open_clip.list_models()`
            device:     Which device to load the model on to
            pretrained: Name of the pretrained checkpoint to load from. See `open_clip.list_pretrained()` for options.
            alpha:      (Optional) used in conjunction with `ckpt_path` to interpolate the checkpoint with pretrained model
            ckpt_path:  (Optional) Path to a saved `.ckpt` file to load the weights from. If not provided, a pretrained model is loaded
            batch_size: Inference batch size

        Some key methods:
            `self.eval_cinemanet()`
            `self.eval_imagenet()`
            `self.eval_shotdeck_clip_datasets()`
            `self.get_image_embeddings()`
        """
        self.arch = arch
        self.device = device
        self._alpha = alpha
        self.pretrained = pretrained
        self.ckpt_path = ckpt_path
        self.batch_size = batch_size

        self._check_args()
        self.model = self.load_model(self.alpha)
        self.tokenizer = self.load_tokenizer()

    @classmethod
    def from_ckpt_path(
        cls, ckpt_path, device, alpha, batch_size, path_embedding, experiment_name
    ):
        ckpt_path = Path(ckpt_path)
        assert ckpt_path.parent.name == "checkpoints"
        root_dir = ckpt_path.parent.parent
        cfg = _load_params_txt(root_dir / "params.txt")

        arch = cfg["model"]
        pretrained = cfg["pretrained"]

        return cls(
            arch=arch, device=device, alpha=alpha, pretrained=pretrained, ckpt_path=ckpt_path,
            batch_size=batch_size, path_embedding=path_embedding, experiment_name=experiment_name,
        )

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha_value):
        self._alpha = alpha_value
        self.model = self.load_model(alpha_value)

    def _check_args(self):
        if self.pretrained and self.ckpt_path:
            if self.alpha is None:
                raise ValueError(
                    f"You provided a pretrained model & checkpoint path to blend, but did not "
                    f"provide alpha values to do the blending"
                )

        if self.ckpt_path is None:
            if self.alpha is not None:
                raise ValueError(
                    f"Attempting to load pretrained model but `alpha` value is also given. Set it to None if you'd like to proceed"
                )

    def load_model(self, alpha: Optional[float]):
        if self.ckpt_path is not None:
            model = load_interpolated_model(
                self.arch, self.device, self.pretrained, self.ckpt_path, alpha=alpha)
            logger.info(f"Loaded interpolated weights at Alpha={self.alpha}")

        else:
            assert self.ckpt_path is None
            assert self.alpha is None
            model = load_model(
                self.arch, self.device, self.pretrained, self.ckpt_path)
            logger.info(f"Loaded pretrained model")

        return model

    def load_tokenizer(self):
        return get_tokenizer(self.arch)

    def _get_eval_args(self, batch_size, num_workers, path_imagenet, wandb_id, device):
        return SimpleNamespace(
            # Model
            model = self.arch,
            custom_text_encoder = None,
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

    def get_image_embeddings(
        self,
        img_files: Optional[List[PathLike]] = None,
        img_folders: Optional[List[PathLike]] = None,
        batch_size: Optional[int] = 16,
        num_workers: int = 4,
    ) -> pd.DataFrame:
        """
        If we have passed `path_embeddings` on init, reads the cached embeddings and returns them
        Else, we compute embeddings, save them to disk, and return them
        """
        if img_files is None and img_folders is None:
            raise RuntimeError(f"Enter `img_files` and/or `img_folders` to point to images that must be analysed")

        return compute_image_embeddings(
            self.model, img_files, img_folders, batch_size or self.batch_size, num_workers)

    def _create_save_embedding_filepath(
        self,
        suffix: Optional[str] = None,
        save_dir = "/home/synopsis/datasets/serialised-datasets/CLIP/"
    ) -> Path:

        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        filename = f"{self.arch}--{self.pretrained}"
        if self.ckpt_path:
            filename = f"{filename}--finetuned-alpha-{str(self.alpha)}"

        if suffix: filename = f"{filename}__{suffix}.feather"
        else:      filename = f"{filename}.feather"

        return save_dir / filename

    def save_embedding_df(
        self,
        df: pd.DataFrame,
        suffix: Optional[str] = None,
        save_dir = "/home/synopsis/datasets/serialised-datasets/CLIP/CLIP-Embeddings-Cached/"
    ) -> PathLike:
        save_path = self._create_save_embedding_filepath(suffix, save_dir)
        df.to_feather(save_path)
        logger.success(f"Saved cached embeddings to {save_path}")

        return save_path

    def eval_imagenet(
        self,
        path_imagenet = "/home/synopsis/datasets/ImageNet/validation/",
        wandb_id = None,
    ) -> Dict[str, float]:
        """
        Run inference on the ImageNet validation set using open-clip's prompts
        """
        from training.train import evaluate
        from training.data import get_imagenet
        from open_clip.factory import image_transform

        args = self._get_eval_args(
            self.batch_size, 4, path_imagenet, wandb_id, self.device)

        imagenet_data = {}
        image_mean = self.model.visual.image_mean
        image_std = self.model.visual.image_std
        preproc = image_transform(
            self.model.visual.image_size, False, image_mean, image_std)
        imagenet_data["imagenet-val"] = get_imagenet(args, (None, preproc), split="val")

        return evaluate(self.model, imagenet_data, 1, args)

    def eval_cinemanet(self, categories: Optional[List[str]] = None) -> Tuple[dict, dict, dict, dict]:
        """
        Run inference on the CinemaNet validation sets

        Returns 4 dictionaries:
            1. Dict[str, float]            -- overall accuracy by category
            2. Dict[str, Dict[str, float]] -- accuracy per label per category
            3. Dict[str, Dict[str, float]] -- inaccuracy per label per category
            4. Dict[str, mpl.Figure]       -- confusion matrices per category
        """
        accuracies_overall = {}
        accuracies_per_label = {}
        inaccuracies_per_label = {}
        confusion_matrices = {}

        categories = categories or DEFAULT_CINEMANET_CATEGORIES
        for category in categories:
            if self.arch.endswith("-custom-text"):
                taxonomy         = {category: TAXONOMY[category]}
                reverse_taxonomy = {category: REVERSE_TAXONOMY[category]}
            else:
                taxonomy         = {category: TAXONOMY_CUSTOM_TOKENS[category]}
                reverse_taxonomy = {category: REVERSE_TAXONOMY_CUSTOM_TOKENS[category]}

            # TODO: Log confusion matrix
            acc,_,_,acc_per_label,inacc_per_label,confusion_matrix = run_cinemanet_eval_by_category(
                self.model, self.tokenizer, category, batch_size=self.batch_size,
                verbose=False, taxonomy=taxonomy, reverse_taxonomy=reverse_taxonomy,
            )
            accuracies_overall[category] = acc
            accuracies_per_label[category] = acc_per_label
            inaccuracies_per_label[category] = inacc_per_label
            confusion_matrices[category] = confusion_matrix

        return accuracies_overall, accuracies_per_label, inaccuracies_per_label, confusion_matrices

    def eval_shotdeck_clip_datasets(self, categories: Optional[List[str]] = None):
        accuracies_overall = {}
        accuracies_per_label = {}
        inaccuracies_per_label = {}
        confusion_matrices = {}

        dh = DataHandlerCLIPValidation()
        DEFAULT_SHOTDECK_CLIP_CATEGORIES = sorted(dh.categories)
        categories = categories or DEFAULT_SHOTDECK_CLIP_CATEGORIES

        for category in categories:
            try:
                acc,_,_,acc_per_label,inacc_per_label,confusion_matrix = run_shotdeck_clip_eval_by_category(
                    self.model, self.tokenizer, category, batch_size=self.batch_size,
                    verbose=False,
                )
                accuracies_overall[category] = acc
                accuracies_per_label[category] = acc_per_label
                inaccuracies_per_label[category] = inacc_per_label
                confusion_matrices[category] = confusion_matrix
            except KeyError:
                pass  # Don't have taxonomy for all CLIP taxonomies yet... derp.

        return accuracies_overall, accuracies_per_label, inaccuracies_per_label, confusion_matrices

    def __repr__(self) -> str:
        path_embed = f"'{self.path_embedding}'" if self.path_embedding else None
        path_ckpt = f"'{self.ckpt_path}'" if self.ckpt_path else None
        return f"""    {self.__class__.__name__}(
                   arch = '{self.arch}',
                 device = {self.device},
                  alpha = {self.alpha},
             pretrained = '{self.pretrained}',
              ckpt_path = {path_ckpt},
         path_embedding = {path_embed},
        experiment_name = '{self.experiment_name}',
    )
"""

def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model

class InferenceModelWhileTraining(InferenceModel):
    def __init__(self, model, tokenizer, orig_state_dict, args, alpha):
        self.model = unwrap_model(model)
        self.model.device = torch.device(args.device)

        if not hasattr(self.model.visual, "image_mean"):
            pretrained_cfg = get_pretrained_cfg(args.model, args.pretrained)
            self.model.visual.image_mean = pretrained_cfg.get('mean', None) or OPENAI_DATASET_MEAN
            self.model.visual.image_std = pretrained_cfg.get('std', None) or OPENAI_DATASET_STD

        self.restore_state_dict = {
            k:v.detach().cpu() for k,v in unwrap_model(model).state_dict().items()}
        self.tokenizer = tokenizer
        self.orig_state_dict = orig_state_dict

        # patch args
        self.args = deepcopy(args)
        self.args.imagenet_val = "/home/synopsis/datasets/ImageNet/validation/"
        self.args.zeroshot_frequency = 1
        self.args.val_frequency = 1
        self.args.distributed = False

        self.batch_size = args.batch_size
        self.device = args.device
        self.ckpt_path = None
        self.pretrained = args.pretrained
        self.arch = args.model

        weights = interpolate_weights(
            {k:v.detach().cpu() for k,v in unwrap_model(self.model).state_dict().items()},
            orig_state_dict,
            alpha,
        )
        unwrap_model(self.model).load_state_dict(weights)

    def _get_eval_args(self, *args, **kwargs):
        return self.args

    def __repr__(self) -> str:
        return f"InferenceModelWhileTraining(...)"
