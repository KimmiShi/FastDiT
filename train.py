# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from utils import set_seed
from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

from torchdistpackage import setup_distributed_slurm, tpc

import optix


#################################################################################
#                             InternEVO                                         #
#################################################################################

import internlm
from internlm.initialize import initialize_distributed_env
from internlm.train import (  # noqa: E402
    get_train_data_loader,
    get_validation_data_loader,
    initialize_llm_profile,
    initialize_model,
    initialize_optimizer,
    record_current_batch_training_metrics)
from internlm.core.context import IS_TENSOR_ZERO_PARALLEL
def set_parallel_attr_for_all(model):
    for param in model.parameters():
        setattr(param, IS_TENSOR_ZERO_PARALLEL, True)
#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def show_mem(msg='', logger=None):
    msg = msg+f' max mem alloc: {torch.cuda.max_memory_allocated()/1024**3:.2f}'
    if logger:
        logger.info(msg)
    else:
        print(msg, flush=True)

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size) #, resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

loss_fn = torch.nn.MSELoss()
def my_loss(output, label):
    return loss_fn(output[:,:4], label)

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    if not args.engine=='evo':
        setup_distributed_slurm()
        set_seed(1024)
    # assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, world_size={dist.get_world_size()}.")

    dp_group = None
    sp_group=None
    if args.sps>1:
        tpc.setup_process_groups([('data', dist.get_world_size()//args.sps), ('sp', args.sps)])
        dp_group = tpc.get_group('data')
        sp_group = tpc.get_group('sp')
    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    # dataset = ImageFolder(args.data_path, transform=transform)
    dataset = torchvision.datasets.CIFAR10(args.data_path, transform=transform, download=True)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(dp_group),
        rank=dist.get_rank(dp_group),
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        dtype=torch.float32 if args.amp else torch.bfloat16,
        sequence_parallel_size=args.sps,
        spg=sp_group,
    ).cuda()
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.grad_ckpt:
        model.enale_grad_ckpt()

    if not args.amp:
        model = model.to(torch.bfloat16)

    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(args.vae).to(device)

    if args.engine == 'optix':
        model, vae, opt, ema = optix.compile(model, vae, use_ema=True, dp_group=dp_group)
    elif args.engine=='evo':
        ema = optix.ShardedEMA(model, group=dp_group)
        opt, beta2_scheduler, lr_scheduler = initialize_optimizer(model=model)
        set_parallel_attr_for_all(model)
        engine = internlm.core.Engine(model, opt, criterion=my_loss)
        engine.train()
    else:
        ema = optix.ShardedEMA(model, group=dp_group)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
        model =DDP(model)


    # Prepare models for training:
    ema.update(model, decay=0)
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    show_mem("before training")
    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    # import pdb;pdb.set_trace()
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        # for i in range(100):
        for iter in loader:
            x, y = iter
            # x = torch.rand([args.batch_size, 3, args.image_size, args.image_size], device='cuda')
            # y = torch.arange(10,10+args.batch_size).cuda()
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            # x = optix.sliced_vae(vae, x)
            if not args.amp:
                x = x.to(torch.bfloat16)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)

            if args.engine=='evo':
                engine.zero_grad()
                outputs = engine(x, t, y)
                loss = engine.criterion(outputs, torch.rand_like(x))
                engine.backward(loss)
                engine.step()
                ema.update(model)

            else:
                with torch.cuda.amp.autocast(enabled=args.amp):
                    loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
                loss.backward()
                opt.step()
                ema.update(model)
                opt.zero_grad()


            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()
                show_mem()


            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                # ema_states = ema.state_dict()
                if rank == 0:
                    checkpoint = {
                        # "model": model.module.state_dict(),
                        # "ema": ema_states,
                        # "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    # parser = argparse.ArgumentParser()
    parser=internlm.get_default_parser()

    parser.add_argument("--data-path", type=str, required=False)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    # parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, default="stabilityai/sd-vae-ft-ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--grad-ckpt", action='store_true')
    parser.add_argument("--amp", action='store_true')
    parser.add_argument("--sps", type=int, default=1)
    parser.add_argument("--engine", type=str)

    args = parser.parse_args()

    if args.engine=='evo':

       initialize_distributed_env(config=args.config)

    main(args)
