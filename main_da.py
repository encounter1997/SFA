# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import argparse
import datetime
import json
import random
import time
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model

import wandb


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    # Logging parameters
    parser.add_argument('--wandb', action='store_true', help="turn on wandb for logging")
    # Optimization parameters
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    # * Transformer DA
    parser.add_argument('--mask_domain_enc', action='store_true',
                        help="apply attn_mask on domain query in def attn in encoder")
    parser.add_argument('--mask_domain_dec', action='store_true',
                        help="apply attn_mask on domain query in self attn in decoder")
    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * DA
    parser.add_argument('--hda', default=None, type=int, nargs='+',
                        help="whether to adopt hierarchical domain adaptation (HDA). "
                             "Use layer index 1, 2, 3, 4, 5. The index is shared by both encoder and decoder. "
                             "HDA is turned off when hda is None.")
    parser.add_argument('--cmt', action='store_true', help="use consistent match loss")
    parser.add_argument('--cmt_start_epoch', default=0, type=int)
    # * Gradient Reverse Layer
    parser.add_argument('--grl_log', action='store_true',
                        help="apply log scheduler for eta in grl,please refer to DANN paper for more detail")
    parser.add_argument('--eta',  default=1, type=float,
                        help='initial eta for gradient reverse layer')
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    # * Loss coefficients for DA
    parser.add_argument('--domain_loss_coef', default=None, type=float,
                        help='same coef for token & query loss in both encoder & decoder')
    parser.add_argument('--domain_enc_token_loss_coef', default=1, type=float)
    parser.add_argument('--domain_dec_token_loss_coef', default=1, type=float)
    parser.add_argument('--domain_enc_query_loss_coef', default=0.1, type=float)
    parser.add_argument('--domain_dec_query_loss_coef', default=0.1, type=float)
    parser.add_argument('--cmt_cls_js_loss_coef', default=1, type=float)
    parser.add_argument('--cmt_bbox_l1_loss_coef', default=1, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='city2foggy')
    parser.add_argument('--coco_path', default='data', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default=None, required=True,
                        help='path where to save, empty string for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"

    # set domain_loss_coef
    if args.domain_loss_coef is not None:
        args.domain_enc_token_loss_coef = args.domain_loss_coef
        args.domain_enc_query_loss_coef = args.domain_loss_coef
        args.domain_dec_token_loss_coef = args.domain_loss_coef
        args.domain_dec_query_loss_coef = args.domain_loss_coef

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # log with wandb
    if utils.get_rank() == 0:
        if args.wandb:
            wandb.init(config=args, project="SFA")
            wandb.run.name = '_'.join([
                args.dataset_file, os.path.basename(args.output_dir), 'bs{}'.format(args.batch_size),
                'seed{}'.format(args.seed),
            ])
        else:
            warnings.warn("wandb is turned off")

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # build dataset
    dataset_train = build_dataset(image_set='train_s', args=args)
    dataset_train_t = build_dataset(image_set='train_t', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_train_t = samplers.NodeDistributedSampler(dataset_train_t)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_train_t = samplers.DistributedSampler(dataset_train_t)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_train_t = torch.utils.data.RandomSampler(dataset_train_t)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    batch_sampler_train_t = torch.utils.data.BatchSampler(sampler_train_t, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_train_t = DataLoader(dataset_train_t, batch_sampler=batch_sampler_train_t,
                                     collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                     pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model_without_ddp.named_parameters():
        print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
        # check the resumed model
        if not args.eval:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
            )
    
    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
            sampler_train_t.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, data_loader_train_t,
            optimizer, device, epoch, args.clip_max_norm,
            grl_log=args.grl_log, eta=args.eta,
        )
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epoch
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )

        results = coco_evaluator.coco_eval['bbox'].stats
        if utils.get_rank() == 0 and args.wandb:
            info = {
                'Average Precision(AP) @ [IoU = 0.50:0.95 | area = all | maxDets = 100]': results[0],
                'Average Precision(AP) @ [IoU = 0.50 | area = all | maxDets = 100]': results[1],
                'Average Precision(AP) @ [IoU = 0.75 | area = all | maxDets = 100]': results[2],
                'Average Precision(AP) @ [IoU = 0.50:0.95 | area = small | maxDets = 100]': results[3],
                'Average Precision(AP) @ [IoU = 0.50:0.95 | area = medium | maxDets = 100]': results[4],
                'Average Precision(AP) @ [IoU = 0.50:0.95 | area = large | maxDets = 100]': results[5],
                'Average Recall(AR) @ [IoU = 0.50:0.95 | area = all | maxDets = 1]': results[6],
                'Average Recall(AR) @ [IoU = 0.50:0.95 | area = all | maxDets = 10]': results[7],
                'Average Recall(AR) @ [IoU = 0.50:0.95 | area = all | maxDets = 100]': results[8],
                'Average Recall(AR) @ [IoU = 0.50:0.95 | area = small | maxDets = 100]': results[9],
                'Average Recall(AR) @ [IoU = 0.50:0.95 | area = medium | maxDets = 100]': results[10],
                'Average Recall(AR) @ [IoU = 0.50:0.95 | area = large | maxDets = 100]': results[11],
            }
            wandb.log(info, step=epoch+1)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def get_gpu_queue(thresh=1000, waittime=100):
    import pynvml

    pynvml.nvmlInit()
    world_size = pynvml.nvmlDeviceGetCount()  # always use all cards for training
    all_gpu_queue = list(range(world_size))
    gpu_queue = []
    while len(gpu_queue) < world_size:
        gpu_queue = []
        candidate_gpu = all_gpu_queue
        for index in candidate_gpu:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                if meminfo.used / 1024 / 1024 < thresh:
                    gpu_queue.append(index)
            except Exception:
                pass
        if len(gpu_queue) < world_size:
            print(
                f"Need {world_size} GPUs for DDP Training, but only {len(gpu_queue)} free devices: {gpu_queue}. "
                f"Waiting for Free GPU ......"
            )
            time.sleep(waittime)


if __name__ == '__main__':
    # get_gpu_queue(thresh=1000, waittime=100)
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
