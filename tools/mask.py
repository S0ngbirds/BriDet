# Copyright (c) OpenMMLab. All rights reserved.
"""Visualize Text_IF_Fusion LIF masks with the configured test dataloader.

This script is intended for dual-modal configs. It reuses the test dataloader
from the config, so visible/infrared loading and packing stay identical to the
normal test pipeline.
"""

import argparse
import os
import os.path as osp

import cv2
import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint

from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize Text_IF_Fusion masks from a trained checkpoint.')
    parser.add_argument('config', help='Test config file path.')
    parser.add_argument('checkpoint', help='Checkpoint file path.')
    parser.add_argument(
        '--out-dir',
        default='outputs/tif_masks',
        help='Directory to save mask visualizations.')
    parser.add_argument(
        '--max-samples',
        type=int,
        default=20,
        help='Maximum number of test samples to visualize.')
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='Device used for inference, e.g. cuda:0 or cpu.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options, same format as tools/test.py.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Job launcher. This script is mainly intended for launcher=none.')
    parser.add_argument(
        '--no-overlay',
        action='store_true',
        help='Only save masks/heatmaps; skip overlay images.')
    parser.add_argument(
        '--save-npy',
        action='store_true',
        help='Also save raw float masks as .npy files.')
    return parser.parse_args()


def get_detector(model):
    if is_model_wrapper(model):
        model = model.module
    return model


def tensor_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: tensor_to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [tensor_to_device(v, device) for v in data]
    if isinstance(data, tuple):
        return tuple(tensor_to_device(v, device) for v in data)
    return data


def unwrap_data_samples(data_samples):
    if data_samples is None:
        return []
    if isinstance(data_samples, (list, tuple)):
        return list(data_samples)
    return [data_samples]


def get_metainfo(sample):
    if sample is None:
        return {}
    if hasattr(sample, 'metainfo'):
        return sample.metainfo
    if isinstance(sample, dict):
        return sample
    return {}


def find_image_path(meta):
    for key in ('img_path', 'ori_img_path', 'vis_path', 'vi_path',
                'visible_path', 'rgb_path', 'filename'):
        path = meta.get(key, None)
        if isinstance(path, str) and osp.isfile(path):
            return path
    return None


def safe_stem(meta, sample_idx):
    path = find_image_path(meta)
    if path is not None:
        return osp.splitext(osp.basename(path))[0]
    img_id = meta.get('img_id', None)
    if img_id is not None:
        return str(img_id)
    return f'sample_{sample_idx:06d}'


def read_overlay_image(meta, fallback_shape):
    path = find_image_path(meta)
    if path is not None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            return img

    ori_shape = meta.get('ori_shape', None)
    if ori_shape is not None and len(ori_shape) >= 2:
        h, w = int(ori_shape[0]), int(ori_shape[1])
    else:
        h, w = fallback_shape
    return np.zeros((h, w, 3), dtype=np.uint8)


def get_target_shape(meta, fallback_shape):
    path = find_image_path(meta)
    if path is not None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            return img.shape[:2]

    ori_shape = meta.get('ori_shape', None)
    if ori_shape is not None and len(ori_shape) >= 2:
        return int(ori_shape[0]), int(ori_shape[1])
    return fallback_shape


def normalize_to_uint8(mask):
    mask = np.asarray(mask, dtype=np.float32)
    min_v = float(mask.min())
    max_v = float(mask.max())
    if max_v - min_v < 1e-6:
        return np.zeros_like(mask, dtype=np.uint8)
    return ((mask - min_v) / (max_v - min_v) * 255).astype(np.uint8)


def save_mask_set(mask,
                  out_prefix,
                  target_shape,
                  image=None,
                  save_npy=False):
    h, w = target_shape
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

    if save_npy:
        np.save(out_prefix + '.npy', mask.astype(np.float32))

    mask_u8 = normalize_to_uint8(mask)
    heatmap = cv2.applyColorMap(mask_u8, cv2.COLORMAP_JET)
    cv2.imwrite(out_prefix + '_mask.png', mask_u8)
    cv2.imwrite(out_prefix + '_heatmap.png', heatmap)

    if image is not None:
        if image.shape[:2] != (h, w):
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        overlay = cv2.addWeighted(image, 0.55, heatmap, 0.45, 0)
        cv2.imwrite(out_prefix + '_overlay.png', overlay)


def build_runner(args):
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.load_from = args.checkpoint
    cfg.work_dir = cfg.get(
        'work_dir',
        osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0]))

    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)
    return runner


def main():
    args = parse_args()
    setup_cache_size_limit_of_dynamo()
    os.makedirs(args.out_dir, exist_ok=True)
    vis_out_dir = osp.join(args.out_dir, 'mask_vis')
    ir_out_dir = osp.join(args.out_dir, 'mask_ir')
    os.makedirs(vis_out_dir, exist_ok=True)
    os.makedirs(ir_out_dir, exist_ok=True)

    runner = build_runner(args)
    model = get_detector(runner.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.to(args.device)
    model.eval()

    if not hasattr(model, 'vision_fuser') or model.vision_fuser is None:
        raise RuntimeError('The loaded model has no vision_fuser module.')

    captured_masks = []

    def tif_hook(module, inputs, output):
        mask = getattr(module, '_last_lif_weight_vis', None)
        if mask is not None:
            captured_masks.append(mask.detach().cpu())

    handle = model.vision_fuser.register_forward_hook(tif_hook)

    sample_count = 0
    try:
        dataloader = runner.test_loop.dataloader
        for data_batch in dataloader:
            if sample_count >= args.max_samples:
                break

            captured_masks.clear()
            data_batch = tensor_to_device(data_batch, args.device)
            data_samples = unwrap_data_samples(data_batch.get('data_samples'))

            with torch.no_grad():
                model.test_step(data_batch)

            if not captured_masks:
                print('No Text_IF_Fusion masks captured for this batch.')
                continue

            batch_size = int(captured_masks[0].shape[0])
            for b in range(batch_size):
                if sample_count >= args.max_samples:
                    break

                sample = data_samples[b] if b < len(data_samples) else None
                meta = get_metainfo(sample)
                stem = safe_stem(meta, sample_count)

                first_mask = captured_masks[0][b, 0].numpy()
                fallback_shape = first_mask.shape
                target_shape = get_target_shape(meta, fallback_shape)
                image = None
                if not args.no_overlay:
                    image = read_overlay_image(meta, fallback_shape)

                for level, level_masks in enumerate(captured_masks):
                    mask_vis = level_masks[b, 0].numpy()
                    mask_ir = 1.0 - mask_vis
                    vis_prefix = osp.join(
                        vis_out_dir,
                        f'{sample_count:06d}_{stem}_level_{level}')
                    ir_prefix = osp.join(
                        ir_out_dir,
                        f'{sample_count:06d}_{stem}_level_{level}')
                    save_mask_set(
                        mask_vis,
                        vis_prefix,
                        target_shape=target_shape,
                        image=image,
                        save_npy=args.save_npy)
                    save_mask_set(
                        mask_ir,
                        ir_prefix,
                        target_shape=target_shape,
                        image=image,
                        save_npy=args.save_npy)

                print(f'Saved {len(captured_masks)} mask level(s) for {stem}.')
                sample_count += 1
    finally:
        handle.remove()

    print(f'Done. Visualized {sample_count} sample(s) to {args.out_dir}.')


if __name__ == '__main__':
    main()
