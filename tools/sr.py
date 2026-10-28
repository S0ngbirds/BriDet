# Copyright (c) OpenMMLab. All rights reserved.
"""Visualize intermediate features from SRBranch.

The script reuses the configured dataloader, so dual-modal loading stays the
same as the normal project pipeline. SRBranch is only executed in the detector
training path, so this script calls ``loss_head_inputs_dict`` under no_grad and
temporarily enables the detector-level training flag.
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
        description='Visualize SRBranch intermediate features.')
    parser.add_argument('config', help='Config file path.')
    parser.add_argument('checkpoint', help='Checkpoint file path.')
    parser.add_argument(
        '--out-dir',
        default='outputs/sr_features',
        help='Directory to save SR feature visualizations.')
    parser.add_argument(
        '--max-samples',
        type=int,
        default=20,
        help='Maximum number of samples to visualize.')
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
        '--save-npy',
        action='store_true',
        help='Also save raw sr_feat and sr_output tensors as .npy files.')
    parser.add_argument(
        '--save-channels',
        action='store_true',
        help='Save visualizations for the first N channels of sr_feat.')
    parser.add_argument(
        '--max-channels',
        type=int,
        default=8,
        help='Number of sr_feat channels to save when --save-channels is set.')
    parser.add_argument(
        '--no-overlay',
        action='store_true',
        help='Only save feature maps/heatmaps; skip overlay images.')
    return parser.parse_args()


def get_detector(model):
    if is_model_wrapper(model):
        return model.module
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


def preprocess_data_batch(model, data_batch, training=False):
    if hasattr(model, 'data_preprocessor') and model.data_preprocessor:
        data_batch = model.data_preprocessor(data_batch, training=training)
    return data_batch


def get_batch_inputs(data_batch):
    if isinstance(data_batch, dict):
        return data_batch.get('inputs')
    return getattr(data_batch, 'inputs', None)


def get_batch_data_samples(data_batch):
    if isinstance(data_batch, dict):
        return data_batch.get('data_samples')
    return getattr(data_batch, 'data_samples', None)


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


def normalize_to_uint8(feat_map):
    feat_map = np.asarray(feat_map, dtype=np.float32)
    min_v = float(feat_map.min())
    max_v = float(feat_map.max())
    if max_v - min_v < 1e-6:
        return np.zeros_like(feat_map, dtype=np.uint8)
    return ((feat_map - min_v) / (max_v - min_v) * 255).astype(np.uint8)


def save_feature_map(feat_map,
                     out_prefix,
                     target_shape,
                     image=None):
    h, w = target_shape
    feat_map = cv2.resize(feat_map, (w, h), interpolation=cv2.INTER_LINEAR)
    feat_u8 = normalize_to_uint8(feat_map)
    heatmap = cv2.applyColorMap(feat_u8, cv2.COLORMAP_JET)

    cv2.imwrite(out_prefix + '_map.png', feat_u8)
    cv2.imwrite(out_prefix + '_heatmap.png', heatmap)

    if image is not None:
        if image.shape[:2] != (h, w):
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        overlay = cv2.addWeighted(image, 0.55, heatmap, 0.45, 0)
        cv2.imwrite(out_prefix + '_overlay.png', overlay)


def save_sr_output(sr_output, out_prefix):
    output = np.asarray(sr_output, dtype=np.float32)
    if output.ndim != 3:
        return

    if output.shape[0] == 1:
        cv2.imwrite(out_prefix + '_sr_output.png', normalize_to_uint8(output[0]))
        return

    channel_groups = [(0, min(3, output.shape[0]), 'first3')]
    if output.shape[0] >= 6:
        channel_groups.append((3, 6, 'last3'))

    for start, end, suffix in channel_groups:
        img = output[start:end]
        if img.shape[0] < 3:
            pad = np.zeros((3 - img.shape[0], img.shape[1], img.shape[2]),
                           dtype=img.dtype)
            img = np.concatenate([img, pad], axis=0)
        img = np.transpose(img[:3], (1, 2, 0))
        img = normalize_to_uint8(img)
        cv2.imwrite(out_prefix + f'_sr_output_{suffix}.png', img)


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
        return Runner.from_cfg(cfg)
    return RUNNERS.build(cfg)


def main():
    args = parse_args()
    setup_cache_size_limit_of_dynamo()
    os.makedirs(args.out_dir, exist_ok=True)

    feat_dir = osp.join(args.out_dir, 'sr_feat')
    output_dir = osp.join(args.out_dir, 'sr_output')
    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    runner = build_runner(args)
    model = get_detector(runner.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.to(args.device)
    model.eval()

    if not hasattr(model, 'sr_branch'):
        raise RuntimeError('The loaded model has no sr_branch module.')

    original_use_sr_branch = getattr(model, 'use_sr_branch', False)
    model.use_sr_branch = True

    captured = []

    def sr_hook(module, inputs, output):
        if not isinstance(output, (list, tuple)) or len(output) < 2:
            return
        sr_output, sr_feat = output[:2]
        captured.append({
            'sr_output': sr_output.detach().cpu(),
            'sr_feat': sr_feat.detach().cpu()
        })

    handle = model.sr_branch.register_forward_hook(sr_hook)

    sample_count = 0
    try:
        dataloader = runner.test_loop.dataloader
        for data_batch in dataloader:
            if sample_count >= args.max_samples:
                break

            captured.clear()
            data_batch = tensor_to_device(data_batch, args.device)
            data_batch = preprocess_data_batch(
                model, data_batch, training=False)
            batch_inputs = get_batch_inputs(data_batch)
            data_samples = unwrap_data_samples(
                get_batch_data_samples(data_batch))

            if batch_inputs is None or not data_samples:
                print('Skip a batch without inputs or data_samples.')
                continue

            old_training = model.training
            model.training = True
            try:
                with torch.no_grad():
                    model.loss_head_inputs_dict(batch_inputs, data_samples)
            finally:
                model.training = old_training

            if not captured:
                print('No SRBranch features captured for this batch.')
                continue

            sr_feat = captured[-1]['sr_feat']
            sr_output = captured[-1]['sr_output']
            batch_size = int(sr_feat.shape[0])

            for b in range(batch_size):
                if sample_count >= args.max_samples:
                    break

                sample = data_samples[b] if b < len(data_samples) else None
                meta = get_metainfo(sample)
                stem = safe_stem(meta, sample_count)

                feat = sr_feat[b].numpy()
                fallback_shape = feat.shape[1:]
                target_shape = get_target_shape(meta, fallback_shape)
                image = None
                if not args.no_overlay:
                    image = read_overlay_image(meta, fallback_shape)

                prefix = osp.join(feat_dir, f'{sample_count:06d}_{stem}')
                if args.save_npy:
                    np.save(prefix + '_sr_feat.npy', feat.astype(np.float32))
                    np.save(prefix + '_sr_output.npy',
                            sr_output[b].numpy().astype(np.float32))

                mean_abs = np.abs(feat).mean(axis=0)
                max_abs = np.abs(feat).max(axis=0)
                save_feature_map(
                    mean_abs,
                    prefix + '_mean_abs',
                    target_shape=target_shape,
                    image=image)
                save_feature_map(
                    max_abs,
                    prefix + '_max_abs',
                    target_shape=target_shape,
                    image=image)

                if args.save_channels:
                    num_channels = min(args.max_channels, feat.shape[0])
                    for channel in range(num_channels):
                        save_feature_map(
                            feat[channel],
                            prefix + f'_channel_{channel:03d}',
                            target_shape=target_shape,
                            image=image)

                output_prefix = osp.join(output_dir, f'{sample_count:06d}_{stem}')
                save_sr_output(sr_output[b].numpy(), output_prefix)

                print(f'Saved SR features for {stem}.')
                sample_count += 1
    finally:
        handle.remove()
        model.use_sr_branch = original_use_sr_branch

    print(f'Done. Visualized {sample_count} sample(s) to {args.out_dir}.')


if __name__ == '__main__':
    main()
