from upyog.all import *
from open_clip import get_tokenizer

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


from cinemanet.CLIP.inference import compute_image_embeddings
from cinemanet.CLIP.utils import load_model, load_interpolated_model
from cinemanet.CLIP.mapping import (
    TAXONOMY, TAXONOMY_CUSTOM_TOKENS, REVERSE_TAXONOMY, REVERSE_TAXONOMY_CUSTOM_TOKENS
)
from cinemanet.CLIP.inference import run_image_classification


__all__ = ["ModelCfg", "InferenceModel"]


@dataclass
class ModelCfg:
    arch: str
    device: Union[str, int, torch.device]
    alpha: float
    pretrained: Optional[str]
    ckpt_path: Optional[str]
    # path_embedding: Optional[str]


class InferenceModel:

    def __init__(
        self,
        arch: str,
        device: Union[str, int, torch.device],
        alpha: Optional[float],
        pretrained: Optional[str],
        ckpt_path: Optional[str],
        batch_size: Optional[int] = 8,
        # path_embedding: Optional[str],
    ):
        self.arch = arch
        self.device = device
        self._alpha = alpha
        self.pretrained = pretrained
        self.ckpt_path = ckpt_path
        self.batch_size = batch_size

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
                raise RuntimeError(
                    f"You provided a pretrained model & checkpoint path to blend, but did not "
                    f"provide alpha values to do the blending"
                )

    def load_model(self, alpha: Optional[float]):
        if self.ckpt_path is not None:
            model = load_interpolated_model(
                self.arch, self.device, self.pretrained, self.ckpt_path, alpha=alpha)
            logger.info(f"Loaded interpolated weights at Alpha={self.alpha}")

        else:
            model = load_model(
                self.arch, self.device, self.pretrained, self.ckpt_path)
            logger.info(f"Loaded pretrained model")

        return model

    def load_tokenizer(self):
        return get_tokenizer(self.arch)

    def _create_embedding_filename(self, suffix: str):
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        filename = f"{self.arch}--{self.pretrained}"
        if self.ckpt_path:
            filename = f"{filename}--finetuned-alpha-{str(self.alpha)}"
        filename = f"{filename}__{suffix}"

        return filename

    def compute_image_embeddings(
        self,
        img_files: Optional[List[PathLike]] = None,
        img_folders: Optional[List[PathLike]] = None,
        batch_size: Optional[int] = 16,
        num_workers: int = 4,
    ):
        if img_files is None and img_folders is None:
            raise RuntimeError

        return compute_image_embeddings(
            self.model, img_files, img_folders, batch_size or self.batch_size, num_workers)

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
        image_mean = getattr(self.model.visual, 'image_mean', None)
        image_std = getattr(self.model.visual, 'image_std', None)
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
