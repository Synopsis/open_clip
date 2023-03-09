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
    sys.path.append("/home/synopsis/git/YOLOX-Custom/")
    sys.path.append("/home/synopsis/git/YOLO-CinemaNet/")
    sys.path.append("/home/synopsis/git/icevision/")
    sys.path.append("/home/synopsis/git/labelling-workflows/")
    sys.path.append("/home/synopsis/git/amalgam/")
    sys.path.append("/home/synopsis/git/cinemanet-multitask-classification/")
    sys.path.append("/home/synopsis/git/Synopsis.py/")


from cinemanet.CLIP.inference import compute_image_embeddings, get_top_matches, view_top_matches
from cinemanet.CLIP.inference import (
    EVALUATION_PROMPTS, CELEBRITY_PROMPTS, PROP_PROMPTS, EMOTION_PROMPTS)
from cinemanet.CLIP.utils import load_model, load_interpolated_model
from cinemanet.CLIP.mapping import (
    TAXONOMY, TAXONOMY_CUSTOM_TOKENS, REVERSE_TAXONOMY, REVERSE_TAXONOMY_CUSTOM_TOKENS
)
from cinemanet.CLIP.inference import run_image_classification


__all__ = ["ModelCfg", "InferenceModel", "InferenceModelFromDisk"]


@dataclass
class ModelCfg:
    arch: str
    device: Union[str, int, torch.device]
    alpha: float
    pretrained: Optional[str]
    ckpt_path: Optional[str]
    # path_embedding: Optional[str]


class InferenceMixin:
    def eval_imagenet(
        self,
        path_imagenet = "/home/synopsis/datasets/ImageNet/validation/",
        wandb_id = None,
    ) -> dict:
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

    def eval_cinemanet(self) -> Tuple[dict, dict, dict]:
        """
        Run inference on the CinemaNet validation sets

        Returns 3 dictionaries:
            1. Dict[str, float]            -- overall accuracy by category
            2. Dict[str, Dict[str, float]] -- accuracy per label per category
            3. Dict[str, Dict[str, float]] -- inaccuracy per label per category
        """
        accuracies_overall = {}
        accuracies_per_label = {}
        inaccuracies_per_label = {}

        for category in TAXONOMY.keys():
            if self.arch.endswith("-custom-text"):
                taxonomy         = {category: TAXONOMY[category]}
                reverse_taxonomy = {category: REVERSE_TAXONOMY[category]}
            else:
                taxonomy         = {category: TAXONOMY_CUSTOM_TOKENS[category]}
                reverse_taxonomy = {category: REVERSE_TAXONOMY_CUSTOM_TOKENS[category]}

            acc,_,_,acc_per_label,inacc_per_label = run_image_classification(
                self.model, self.tokenizer, category, batch_size=self.batch_size,
                verbose=False, taxonomy=taxonomy, reverse_taxonomy=reverse_taxonomy,
            )
            accuracies_overall[category] = acc
            accuracies_per_label[category] = acc_per_label
            inaccuracies_per_label[category] = inacc_per_label

        return accuracies_overall, accuracies_per_label, inaccuracies_per_label


class InferenceModelFromDisk(InferenceMixin):

    def __init__(
        self,
        arch: str,
        device: Union[str, int, torch.device],
        alpha: Optional[float],
        pretrained: Optional[str],
        ckpt_path: Optional[str],
        batch_size: Optional[int] = 8,
        # ...
        path_embedding: Optional[PathLike] = None,
        experiment_name: str = None,
    ):
        self.arch = arch
        self.device = device
        self._alpha = alpha
        self.pretrained = pretrained
        self.ckpt_path = ckpt_path
        self.batch_size = batch_size
        self.path_embedding = path_embedding
        assert experiment_name, f"`experiment_name` required"
        self.experiment_name = experiment_name

        self._check_args()
        self.model = self.load_model(self.alpha)
        self.tokenizer = self.load_tokenizer()

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

    def _create_embedding_filename(
        self,
        suffix: str,
        save_dir = "/home/synopsis/datasets/serialised-datasets/CLIP/"
    ) -> Path:

        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        filename = f"{self.arch}--{self.pretrained}"
        if self.ckpt_path:
            filename = f"{filename}--finetuned-alpha-{str(self.alpha)}"
        filename = f"{filename}__{suffix}.feather"

        return save_dir / filename

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
        save_dir: Optional[PathLike] = "/home/synopsis/datasets/serialised-datasets/CLIP/CLIP-Embeddings-Cached/",
    ) -> pd.DataFrame:
        """
        If we have passed `path_embeddings` on init, reads the cached embeddings and returns them
        Else, we compute embeddings, save them to disk, and return them
        """
        if self.path_embedding is not None:
            logger.info(f"Reading pre-computed embeddings")
            return pd.read_feather(self.path_embedding)

        else:
            if save_dir is None:
                raise ValueError(f"Enter a `save_dir` to save the embeddings in")

        save_dir = Path(save_dir)
        if img_files is None and img_folders is None:
            raise RuntimeError(f"Enter `img_files` and/or `img_folders` to point to images that must be analysed")

        df = compute_image_embeddings(
            self.model, img_files, img_folders, batch_size or self.batch_size, num_workers)

        save_path = self._create_embedding_filename(self.experiment_name, save_dir)
        df.to_feather(save_path)
        self.path_embedding = save_path
        logger.success(f"Saved cached embeddings to {save_path}")

        return df

    def eval_cinemanet(self) -> Tuple[dict, dict, dict]:
        self.cnet_acc, self.cnet_acc_per_label, self.cnet_inacc_per_label = super().eval_cinemanet()
        return self.cnet_acc, self.cnet_acc_per_label, self.cnet_inacc_per_label

    def eval_imagenet(self, path_imagenet="/home/synopsis/datasets/ImageNet/validation/", wandb_id=None) -> dict:
        self.imgnet_metrics = super().eval_imagenet(path_imagenet, wandb_id)
        return self.imgnet_metrics

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

class InferenceModel(InferenceMixin):
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

        from cinemanet.CLIP.utils import interpolate_weights

        weights = interpolate_weights(
            {k:v.detach().cpu() for k,v in unwrap_model(self.model).state_dict().items()},
            orig_state_dict,
            alpha,
        )
        unwrap_model(self.model).load_state_dict(weights)

    def _get_eval_args(self, *args, **kwargs):
        return self.args

    def get_image_embeddings(
        self,
        img_files: Optional[List[PathLike]] = None,
        img_folders: Optional[List[PathLike]] = None,
        batch_size: Optional[int] = 16,
        num_workers: int = 4,
    ):
        return compute_image_embeddings(
            self.model, img_files, img_folders, batch_size, num_workers, "Computing ShotDeck Embeddings...")
