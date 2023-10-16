import glob
import logging
import os
import re
import subprocess
import sys
import random
from datetime import datetime

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
from training.data import get_data
from training.distributed import is_master, init_distributed_device, broadcast_object
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr, const_lr, const_lr_cooldown
from training.train import train_one_epoch, evaluate
from training.file_utils import pt_load, check_exists, start_sync_process, remote_sync


LATEST_CHECKPOINT_NAME = "epoch_latest.pt"

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


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote : bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def main(args):
    args = parse_args(args)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([
            date_str,
            f"model_{model_name_safe}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

    resume_latest = args.resume == 'latest'
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
            if args.save_most_recent:
                print('Error. Cannot use save-most-recent with remote_sync and resume latest.')
                return -1
            if args.remote_sync_protocol != 's3':
                print('Error. Sync protocol not supported when using resume latest.')
                return -1
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.copy_codebase:
        copy_codebase(args)

    # start the sync proces if remote-sync is not None
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        # first make sure it works
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('remote sync successful.')
        else:
            logging.info('Error: remote sync failed. Exiting.')
            return -1
        # if all looks good, start a process to do this every args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        remote_sync_process.start()

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    dist_model = None
    args.distill = args.distill_model is not None and args.distill_pretrained is not None
    if args.distill:
        #FIXME: support distillation with grad accum.
        assert args.accum_freq == 1
        #FIXME: support distillation with coca.
        assert 'coca' not in args.model.lower()

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]
    random_seed(args.seed, 0)
    tokenizer = get_tokenizer(args.model, args)
    model_kwargs = {}
    if args.siglip:
        model_kwargs['init_logit_scale'] = np.log(10)  # different from CLIP
        model_kwargs['init_logit_bias'] = -10
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        pretrained_image=args.pretrained_image,
        image_mean=args.image_mean,
        image_std=args.image_std,
        aug_cfg=args.aug_cfg,
        output_dict=True,
        **model_kwargs,
    )
    orig_state_dict = {
        k:v.detach().cpu()
        for k,v in unwrap_model(model).state_dict().items()
    }
    import rich
    rich.print(f"Preprocess Train:\n{preprocess_train}\n")
    rich.print(f"Preprocess Val:\n{preprocess_val}\n")

    if args and args.custom_text_encoder:
        from cinemanet_clip.text_modelling import update_text_encoder

        model = update_text_encoder(model, tokenizer, tokenizer.placeholder_token_ids)
        logging.info(f"Updated text encoder")
    if args.distill:
        # FIXME: currently assumes the model you're distilling from has the same tokenizer & transforms.
        dist_model, _, _ = create_model_and_transforms(
            args.distill_model, 
            args.distill_pretrained,
            device=device,
            precision=args.precision,
            output_dict=True,
        )
    if args.use_bnb_linear is not None:
        print('=> using a layer from bitsandbytes.\n'
              '   this is an experimental feature which requires two extra pip installs\n'
              '   pip install bitsandbytes triton'
              '   please make sure to use triton 2.0.0')
        import bitsandbytes as bnb
        from open_clip.utils import replace_linear
        print(f'=> replacing linear layers with {args.use_bnb_linear}')
        linear_replacement_cls = getattr(bnb.nn.triton_based_modules, args.use_bnb_linear)
        replace_linear(model, linear_replacement_cls)
        model = model.to(device)

    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)
    if args.lock_text:
        model.lock_text_tower(
                        unlocked_layers = args.lock_text_unlocked_layers,
                      freeze_layer_norm = args.lock_text_freeze_layer_norm,
            # unlock_positional_embedding = args.unlock_text_pos_embed,
                #  unlock_token_embedding = args.unlock_text_token_embed,
        )

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
    
        if args.distill:
            dist_model = torch.nn.parallel.DistributedDataParallel(dist_model, device_ids=[device], **ddp_args)

    # create optimizer and scaler
    optimizer = None
    scaler = None

    if args.train_data or args.dataset_type == "synthetic":
        assert not args.trace, 'Cannot train with traced model'

        # NOTE: I (Rahul) have verified outside the script that no parts of the
        # text/positional embedding get excluded by the `exclude` function as
        # there's no BN, LN, bias, or logit_scale parameters in these weights
        # def exclude(n, p) -> bool:
        #     to_exclude = p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        #     if to_exclude and n=="positional_embedding":
        #         print("WARNING!! Excluding Parts Of Positional Text Embedding!")

        #     if to_exclude and n=="token_embedding.weight":
        #         print("WARNING!! Excluding Parts Of Token Text Embedding!")

        #     return to_exclude

        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)

        # if args.unlock_text_pos_embed or args.unlock_text_token_embed:
        #     non_tok_pos_embedding_params = []
        #     _found_pos = False
        #     _found_tok = False
        #     for (name, params) in model.named_parameters():
        #         is_pos_embedding = name == "positional_embedding"
        #         is_token_embedding = name == "token_embedding.weight"

        #         if is_pos_embedding or is_token_embedding:
        #             if is_pos_embedding:
        #                 _found_pos = True
        #                 pos_embedding_params = [(name, params)]
        #             if is_token_embedding:
        #                 _found_tok = True
        #                 token_embedding_params = [(name, params)]

        #         else:
        #             non_tok_pos_embedding_params.append( (name, params) )

        #     assert _found_pos, f"Failed to find positional text embedding in model params"
        #     assert _found_tok, f"Failed to find token text embedding in model params"

        #     gain_or_bias_params = [p for n, p in non_tok_pos_embedding_params if exclude(n, p) and p.requires_grad]
        #     rest_params = [p for n, p in non_tok_pos_embedding_params if include(n, p) and p.requires_grad]

        #     pos_embedding_trainable_params = [p for n,p in pos_embedding_params if p.requires_grad]
        #     token_embedding_trainable_params = [p for n,p in token_embedding_params if p.requires_grad]

        #     optimizer = optim.AdamW(
        #         [
        #             {"params": gain_or_bias_params, "weight_decay": 0., "name": "bn_ln_bias_logit_scale_params"},
        #             {"params": rest_params, "weight_decay": args.wd,    "name": "model_params"},
        #             {"params": pos_embedding_trainable_params,   "weight_decay": args.wd, "lr": args.lr_text_pos_embed, "name": "positional_text_embedding"},
        #             {"params": token_embedding_trainable_params, "weight_decay": args.wd, "lr": args.lr_text_token_embed, "name": "token_text_embedding"},
        #         ],
        #         lr=args.lr,
        #         betas=(args.beta1, args.beta2),
        #         eps=args.eps,
        #     )

        # else:
        #     named_parameters = list(model.named_parameters())
        #     gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        #     rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        #     optimizer = optim.AdamW(
        #         [
        #             {"params": gain_or_bias_params, "weight_decay": 0., "name": "bn_ln_bias_logit_scale_params"},
        #             {"params": rest_params, "weight_decay": args.wd,    "name": "model_params"},
        #         ],
        #         lr=args.lr,
        #         betas=(args.beta1, args.beta2),
        #         eps=args.eps,
        #     )

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        optimizer = optim.AdamW(
                [
                    {"params": gain_or_bias_params, "weight_decay": 0., "name": "bn_ln_bias_logit_scale_params"},
                    {"params": rest_params, "weight_decay": args.wd,    "name": "model_params"},
                ],
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
            )

        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = GradScaler() if args.precision == "amp" else None

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    if args.ckpt_cinemaclip_for_text_enc:
        model.load_nonvisual_weights_from_clip_ckpt(ckpt_path=args.ckpt_cinemaclip_for_text_enc)

    if args.ckpt_cinemanet_for_img_enc:
        model.load_visual_weights_from_cinemanet_ckpt(ckpt_path=args.ckpt_cinemanet_for_img_enc)

    # initialize datasets
    data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch, tokenizer=tokenizer)
    assert len(data), 'At least one train or eval dataset must be specified.'

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None,\
                "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer, args.lr, args.warmup, total_steps,
                cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
        else:
            logging.error(
                f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
            exit(1)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        img_size: int = unwrap_model(model).visual.image_size[0]
        wandb_name = args.wandb_run_name or f"{model_name_safe}__imgsz-{img_size}__num-train-lyr-{args.lock_image_unlocked_groups}"
        wandb_name = wandb_name + f"__{date_str}"
        wandb.init(
            project=args.wandb_project_name,
            name=wandb_name,
            id=args.name,
            notes=args.wandb_notes,
            tags=args.wandb_tags, 
            resume='auto' if args.resume == "latest" else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    original_model = model
    if args.torchcompile:
        logging.info('Compiling model...')
        model = torch.compile(original_model)

    if 'train' not in data:
        # If using int8, convert to inference mode.
        if args.use_bnb_linear is not None:
            from open_clip.utils import convert_int8_model_to_inference_mode
            convert_int8_model_to_inference_mode(model)
        # Evaluate.
        evaluate(model, data, start_epoch, args, writer)
        return

    loss = create_loss(args)

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=writer)
        completed_epoch = epoch + 1

        # Saving checkpoints.
        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": original_model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                save_path = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt")
                torch.save(checkpoint_dict, save_path)

                restore_state_dict = {
                    k:v.detach().cpu() for k,v in unwrap_model(model).state_dict().items()}

                if is_master(args):
                    from inference import InferenceModelWhileTraining
                    alphas = args.val_alphas
                    if completed_epoch == 1 and args.val_alpha_0:
                        alphas.insert(0, 0.0)

                    for alpha in alphas:
                        metrics = {}
                        unwrap_model(model).load_state_dict(restore_state_dict)
                        inf = InferenceModelWhileTraining(model, tokenizer, orig_state_dict, args, alpha)

                        # Run evaluation on ImageNet + CinemaNet + ShotDeck Datasets
                        imgnet_metrics = inf.eval_imagenet()
                        cinemanet_metrics, _, _, cinemanet_confusion_matrices = inf.eval_cinemanet(
                            categories = args.cinemanet_eval_categories,
                            viz_title = wandb_name,
                        )
                        mean_cnet_acc = sum(cinemanet_metrics.values()) / len(cinemanet_metrics)
                        # shotdeck_clip_metrics, _, _, shotdeck_clip_confusion_matrices = inf.eval_shotdeck_clip_datasets()

                        # Log Metrics
                        metrics.update(imgnet_metrics)
                        metrics.update({f"{k}-zeroshot-val-top1": v for k,v in cinemanet_metrics.items()})
                        # metrics.update({f"shotdeck-{k}-zeroshot-val-top1": v for k,v in shotdeck_clip_metrics.items()})
                        metrics.update({"cinemanet-all_categories-avg-val-top1": mean_cnet_acc})

                        logging.info(
                            f"Eval Epoch: {epoch} "
                            + "\t".join([f"{k} (alpha={alpha}): {round(v, 4):.4f}" for k, v in metrics.items()])
                        )

                        for name, val in metrics.items():
                            wandb.log({f"val/alpha={alpha}/{name}": val, 'epoch': completed_epoch, 'alpha': alpha})
                            # wandb.log({f"val/{name}": val, 'epoch': completed_epoch, 'alpha': alpha})

                        # Log Confusion Matrices
                        wandb_confusion_matrices = {}
                        for (category, conf_matrix) in cinemanet_confusion_matrices.items():
                            key = f"val/alpha={alpha}/{category}-confusion_matrix"
                            value = wandb.Image(conf_matrix)
                            wandb_confusion_matrices[key] = value

                        # for (category, conf_matrix) in shotdeck_clip_confusion_matrices.items():
                        #     key = f"val/alpha={alpha}/shotdeck-{category}-confusion_matrix"
                        #     value = wandb.Image(conf_matrix)
                        #     wandb_confusion_matrices[key] = value
                        wandb.log(wandb_confusion_matrices)

                unwrap_model(model).load_state_dict(restore_state_dict)


            if args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)

            if args.save_most_recent:
                # try not to corrupt the latest checkpoint if save fails
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)

    if args.wandb and is_master(args) and args.subjective_eval_file:
        from cinemanet_clip.inference import (
            get_top_matches, view_top_matches,
            EVALUATION_PROMPTS, CELEBRITY_PROMPTS, PROP_PROMPTS)
        from inference import InferenceModelWhileTraining
        from upyog.all import load_json, tqdm, Path
        from synopsis_labelling.utils.file_fetching import find_shotdeck_thumb_file_on_disk

        restore_state_dict = {
            k:v.detach().cpu() for k,v in unwrap_model(model).state_dict().items()}

        model.eval()
        pbar = tqdm([0.0, 0.5, 0.75, 1.0])
        N = 21
        for alpha in pbar:
            pbar.set_description(f"Evaluating @ α={alpha}")
            base_log_key = f"val/prompt_matches/alpha={alpha}"
            log_images = {}

            # Create model
            unwrap_model(model).load_state_dict(restore_state_dict)
            inf = InferenceModelWhileTraining(model, tokenizer, orig_state_dict, args, alpha)

            # Get embeddings
            filepaths = load_json(args.subjective_eval_file)
            filepaths = [find_shotdeck_thumb_file_on_disk(Path(f).name) for f in filepaths]
            df = inf.get_image_embeddings(filepaths, batch_size=args.batch_size, num_workers=args.workers)
            embeddings = np.stack(df.embedding)

            # Eval prompts
            for prompts, prompt_category in [
                (EVALUATION_PROMPTS, "Evaluation prompts"),
                (CELEBRITY_PROMPTS, "Celebrity prompts"),
                (PROP_PROMPTS, "Prop prompts"),
            ]:
                for prompt in tqdm(prompts, prompt_category):
                    top_matches = get_top_matches(prompt, inf.model, embeddings, inf.tokenizer)
                    grid = view_top_matches(df, top_matches, N)

                    log_images[f"{base_log_key}/{prompt_category}/{prompt}"] = grid
            log_images = {k: wandb.Image(v) for k,v in log_images.items()}
            wandb.log(log_images)

        wandb.finish()

    # run a final sync.
    if remote_sync_process is not None:
        logging.info('Final remote sync.')
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('Final remote sync successful.')
        else:
            logging.info('Final remote sync failed.')
    

def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main(sys.argv[1:])
