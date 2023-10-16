""" timm model adapter

Wraps timm (https://github.com/rwightman/pytorch-image-models) models for use as a vision tower in CLIP model.
"""
import logging
from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn

try:
    import timm
    from timm.models.layers import Mlp, to_2tuple
    try:
        # old timm imports < 0.8.1
        from timm.models.layers.attention_pool2d import RotAttentionPool2d
        from timm.models.layers.attention_pool2d import AttentionPool2d as AbsAttentionPool2d
    except ImportError:
        # new timm imports >= 0.8.1
        from timm.layers import RotAttentionPool2d
        from timm.layers import AttentionPool2d as AbsAttentionPool2d
except ImportError:
    timm = None

from .utils import freeze_batch_norm_2d


# NOTE: Copied over from cinemanet-multitas-classification pkg for brevity
# Removed dropout layers (following timm practices for open-clip)
# The main diff here is that `hidden_dim` is hard-coded to 512. As per timm open-clip implementation,
# it should be (640*2 = 1280)
def _create_mlp_proj_head(
    in_feat=1280, out_feat=640, hidden_dim=512
) -> nn.Sequential:
    return nn.Sequential(
        nn.Dropout(0.0),
        nn.Linear(in_feat, hidden_dim, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(hidden_dim, eps=1e-5),
        nn.Dropout(0.0),
        nn.Linear(hidden_dim, out_feat)
    )


class TimmModel(nn.Module):
    """ timm model adapter
    """

    def __init__(
            self,
            model_name,
            embed_dim,
            image_size=224,
            pool='avg',
            proj='linear',
            proj_bias=False,
            drop=0.,
            drop_path=None,
            patch_drop=None,
            pretrained=False,
    ):
        super().__init__()
        if timm is None:
            raise RuntimeError("Please `pip install timm` to use timm models.")
        self.image_size = to_2tuple(image_size)
        self.proj = proj

        # setup kwargs that may not be common across all models
        timm_kwargs = {}
        if "ghostnet" in model_name:
            timm_kwargs['drop_rate'] = drop_path
        else:
            if drop_path is not None:
                timm_kwargs['drop_path_rate'] = drop_path
            if patch_drop is not None:
                timm_kwargs['patch_drop_rate'] = patch_drop
        # if drop_path is not None:
        #     timm_kwargs['drop_path_rate'] = drop_path
        # if patch_drop is not None:
        #     timm_kwargs['patch_drop_rate'] = patch_drop

        custom_pool = pool in ('abs_attn', 'rot_attn')
        if not proj and not custom_pool:
            # use network classifier head as projection if no proj specified and no custom pooling used
            self.trunk = timm.create_model(
                model_name,
                num_classes=embed_dim,
                global_pool=pool,
                pretrained=pretrained,
                **timm_kwargs,
            )
            prev_chs = embed_dim
        else:
            self.trunk = timm.create_model(
                model_name,
                pretrained=pretrained,
                **timm_kwargs,
            )
            feat_size = self.trunk.default_cfg.get('pool_size', None)
            feature_ndim = 1 if not feat_size else 2
            if custom_pool:
                assert feature_ndim == 2
                # if attn pooling used, remove both classifier and default pool
                self.trunk.reset_classifier(0, global_pool='')
            else:
                # reset global pool if pool config set, otherwise leave as network default
                reset_kwargs = dict(global_pool=pool) if pool else {}
                self.trunk.reset_classifier(0, **reset_kwargs)
            prev_chs = self.trunk.num_features

        head_layers = OrderedDict()

        # Add custom pooling to head
        if pool == 'abs_attn':
            head_layers['pool'] = AbsAttentionPool2d(prev_chs, feat_size=feat_size, out_features=embed_dim)
            prev_chs = embed_dim
        elif pool == 'rot_attn':
            head_layers['pool'] = RotAttentionPool2d(prev_chs, out_features=embed_dim)
            prev_chs = embed_dim

        # NOTE attention pool ends with a projection layer, so proj should usually be set to '' if such pooling is used
        if proj == 'linear':
            head_layers['drop'] = nn.Dropout(drop)
            head_layers['proj'] = nn.Linear(prev_chs, embed_dim, bias=proj_bias)
        elif proj == 'mlp':
            head_layers['mlp'] = Mlp(prev_chs, 2 * embed_dim, embed_dim, drop=(drop, 0), bias=(True, proj_bias))
        elif proj == 'mlp_cinemanet':
            head_layers['mlp'] = _create_mlp_proj_head(prev_chs, hidden_dim=512, out_feat=embed_dim)
        else:
            assert not proj, f'Unknown projection type {proj}.'

        self.head = nn.Sequential(head_layers)

    def load_cinemanet_backbone_checkpoint(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cnet_weights = ckpt["state_dict"]
        weights_renamed = OrderedDict()

        for k,v in cnet_weights.items():
            # Irrelevant keys
            if "classifier_heads" in k: continue
            if "loss_functions" in k:   continue

            # Copy over relevant keys, renaming them as needed
            if "model.encoder" in k:
                k_target = k.replace("model.encoder.", "trunk.")
            elif "embed_proj_head" in k:
                k_target = k.replace("model.embed_proj_head.", "head.mlp.")
            else:
                raise RuntimeError(f"Unexpected key {k}")

            weights_renamed[k_target] = v

        missing = self.load_state_dict(weights_renamed, strict=False)
        # If using CinemaNet mlp head explicitly, we must have loaded in pre-trained adapter weights
        if self.proj == "mlp_cinemanet":
            assert missing.unexpected_keys == []
            assert missing.missing_keys == []
        # Else, we may or may not have loaded pre-trained adapter weights.
        # As of 16 Oct 2023, we certainly do not. However, in the future, we may have the same components
        # adapted into the backbone training and will be loading pre-trained weights here too.
        else:
            if len(missing.missing_keys) > 0:
                for key in missing.missing_keys: assert "head." in key
                print(f"Missing following keys from ckpt for adapter head: {missing.missing_keys}")
                print(f"Training a freshly initialised non pre-trained adapter")

        return missing


    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        """ lock modules
        Args:
            unlocked_groups (int): leave last n layer groups unlocked (default: 0)
        """
        if not unlocked_groups:
            # lock full model
            for param in self.trunk.parameters():
                param.requires_grad = False
            if freeze_bn_stats:
                freeze_batch_norm_2d(self.trunk)
        else:
            # NOTE: partial freeze requires latest timm (master) branch and is subject to change
            try:
                # FIXME import here until API stable and in an official release
                from timm.models.helpers import group_parameters, group_modules
            except ImportError:
                raise RuntimeError(
                    'Please install latest timm `pip install git+https://github.com/rwightman/pytorch-image-models`')
            matcher = self.trunk.group_matcher()
            gparams = group_parameters(self.trunk, matcher)
            max_layer_id = max(gparams.keys())
            max_layer_id = max_layer_id - unlocked_groups
            for group_idx in range(max_layer_id + 1):
                group = gparams[group_idx]
                for param in group:
                    self.trunk.get_parameter(param).requires_grad = False
            if freeze_bn_stats:
                gmodules = group_modules(self.trunk, matcher, reverse=True)
                gmodules = {k for k, v in gmodules.items() if v <= max_layer_id}
                freeze_batch_norm_2d(self.trunk, gmodules)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        try:
            self.trunk.set_grad_checkpointing(enable)
        except Exception as e:
            logging.warning('grad checkpointing not supported for this timm image tower, continuing without...')

    def forward(self, x):
        x = self.trunk(x)
        x = self.head(x)
        return x
