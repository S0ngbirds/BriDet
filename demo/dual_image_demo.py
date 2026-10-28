# Copyright (c) OpenMMLab. All rights reserved.
"""Dual image demo for paired visible/infrared inputs."""

import ast
import copy
import logging
import os
import os.path as osp
from argparse import ArgumentParser

import cv2
import mmcv
import mmengine
import numpy as np
from mmengine.dataset import Compose
from mmengine.logging import print_log

from mmdet.apis import DetInferencer
from mmdet.evaluation import get_classes


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')

YOLO_COLORS = (
    (255, 56, 56),
    (255, 157, 151),
    (255, 112, 31),
    (255, 178, 29),
    (207, 210, 49),
    (72, 249, 10),
    (146, 204, 23),
    (61, 219, 134),
    (26, 147, 52),
    (0, 212, 187),
    (44, 153, 168),
    (0, 194, 255),
    (52, 69, 147),
    (100, 115, 255),
    (0, 24, 236),
    (132, 56, 255),
    (82, 0, 133),
    (203, 56, 255),
    (255, 149, 200),
    (255, 55, 199),
)


class DualImageDetInferencer(DetInferencer):
    """DetInferencer that visualizes paired-path inputs on the first image."""

    def _init_pipeline(self, cfg):
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline

        # For inference, the key of ``img_id`` is not used.
        if 'meta_keys' in pipeline_cfg[-1]:
            pipeline_cfg[-1]['meta_keys'] = tuple(
                meta_key for meta_key in pipeline_cfg[-1]['meta_keys']
                if meta_key != 'img_id')

        return Compose(pipeline_cfg)

    def pred2dict(self, data_sample, pred_out_dir=''):
        img_path = getattr(data_sample, 'img_path', None)
        if isinstance(img_path, (list, tuple)):
            data_sample.set_metainfo({'img_path': img_path[0]})
            try:
                return super().pred2dict(data_sample, pred_out_dir)
            finally:
                data_sample.set_metainfo({'img_path': img_path})
        return super().pred2dict(data_sample, pred_out_dir)

    def visualize(self,
                  inputs,
                  preds,
                  return_vis=False,
                  show=False,
                  wait_time=0,
                  draw_pred=True,
                  pred_score_thr=0.3,
                  no_save_vis=False,
                  img_out_dir='',
                  **kwargs):
        if no_save_vis is True:
            img_out_dir = ''

        if not show and img_out_dir == '' and not return_vis:
            return None

        if self.visualizer is None:
            raise ValueError('Visualization needs the "visualizer" term '
                             'defined in the config, but got None.')

        results = []
        for single_input, pred in zip(inputs, preds):
            if isinstance(single_input, (list, tuple)):
                vis_input = single_input[0]
            else:
                vis_input = single_input

            if isinstance(vis_input, str):
                img_bytes = mmengine.fileio.get(vis_input)
                img = mmcv.imfrombytes(img_bytes)
                img = img[:, :, ::-1]
                img_name = osp.basename(vis_input)
            elif isinstance(vis_input, np.ndarray):
                img = vis_input.copy()
                img_num = str(self.num_visualized_imgs).zfill(8)
                img_name = f'{img_num}.jpg'
            else:
                raise ValueError(f'Unsupported input type: {type(vis_input)}')

            out_file = osp.join(img_out_dir, 'vis',
                                img_name) if img_out_dir != '' else None

            drawn_img = _draw_yolo_predictions(
                img,
                pred,
                self.visualizer.dataset_meta.get('classes', None),
                pred_score_thr,
                draw_pred,
            )
            self.visualizer.set_image(drawn_img)

            if show:
                self.visualizer.show(
                    drawn_img, win_name=img_name, wait_time=wait_time)

            if out_file is not None:
                mmengine.mkdir_or_exist(osp.dirname(out_file))
                mmcv.imwrite(drawn_img[..., ::-1], out_file)
            else:
                self.visualizer.add_image(
                    img_name, drawn_img, self.num_visualized_imgs)

            results.append(drawn_img)
            self.num_visualized_imgs += 1

        return results


def _to_numpy(data):
    if hasattr(data, 'tensor'):
        data = data.tensor
    if hasattr(data, 'detach'):
        data = data.detach()
    if hasattr(data, 'cpu'):
        data = data.cpu()
    if hasattr(data, 'numpy'):
        data = data.numpy()
    return np.asarray(data)


def _get_label_name(instances, classes, index, label):
    if 'label_names' in instances:
        return str(instances.label_names[index])
    if classes is not None and 0 <= label < len(classes):
        return str(classes[label])
    return f'class {label}'


def _draw_yolo_predictions(img, pred, classes, pred_score_thr, draw_pred=True):
    """Draw YOLO-style boxes on an RGB image."""
    drawn_img = img.clip(0, 255).astype(np.uint8).copy()
    if not draw_pred or pred is None or 'pred_instances' not in pred:
        return drawn_img

    instances = pred.cpu().pred_instances
    if 'bboxes' not in instances or 'scores' not in instances:
        return drawn_img

    keep = instances.scores > pred_score_thr
    instances = instances[keep]
    if len(instances) == 0:
        return drawn_img

    bboxes = _to_numpy(instances.bboxes)
    scores = _to_numpy(instances.scores)
    labels = _to_numpy(instances.labels).astype(np.int64)
    height, width = drawn_img.shape[:2]

    line_width = max(round((height + width) * 0.0015), 1)
    font_scale = max(line_width / 4, 0.35)
    text_thickness = max(line_width - 1, 1)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, (bbox, score, label) in enumerate(zip(bboxes, scores, labels)):
        x1, y1, x2, y2 = bbox[:4]
        x1 = int(np.clip(round(x1), 0, width - 1))
        y1 = int(np.clip(round(y1), 0, height - 1))
        x2 = int(np.clip(round(x2), 0, width - 1))
        y2 = int(np.clip(round(y2), 0, height - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        color = YOLO_COLORS[int(label) % len(YOLO_COLORS)]
        label_name = _get_label_name(instances, classes, i, int(label))
        caption = f'{label_name} {float(score):.2f}'

        cv2.rectangle(drawn_img, (x1, y1), (x2, y2), color, line_width)

        (text_w, text_h), baseline = cv2.getTextSize(
            caption, font, font_scale, text_thickness)
        outside = y1 - text_h - baseline - 3 >= 0
        label_y1 = y1 - text_h - baseline - 3 if outside else y1
        label_y2 = y1 if outside else y1 + text_h + baseline + 3
        label_x2 = min(x1 + text_w + 4, width - 1)

        cv2.rectangle(
            drawn_img, (x1, label_y1), (label_x2, label_y2), color, -1)
        text_y = y1 - baseline - 2 if outside else y1 + text_h + 1
        cv2.putText(
            drawn_img,
            caption,
            (x1 + 2, text_y),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            lineType=cv2.LINE_AA)

    return drawn_img


def _is_image(path):
    return osp.splitext(path)[1].lower() in IMG_EXTENSIONS


def _list_images(path):
    return [
        osp.join(path, name) for name in sorted(os.listdir(path))
        if _is_image(name) and osp.isfile(osp.join(path, name))
    ]


def _index_images_by_stem(path):
    images = _list_images(path)
    return {
        osp.splitext(osp.basename(img_path))[0]: img_path
        for img_path in images
    }


def _infer_ir_path(vis_path):
    parts = list(osp.normpath(vis_path).split(os.sep))
    for i, part in enumerate(parts):
        if part.lower() == 'vis':
            parts[i] = 'ir'
            return os.sep.join(parts)
    root, name = osp.split(vis_path)
    return osp.join(osp.dirname(root), 'ir', name)


def _build_pairs(vis_inputs, ir_inputs=None):
    if osp.isdir(vis_inputs):
        vis_paths = _list_images(vis_inputs)
        if ir_inputs is None:
            ir_root = _infer_ir_path(vis_inputs)
        else:
            ir_root = ir_inputs
        if not osp.isdir(ir_root):
            raise FileNotFoundError(f'IR folder not found: {ir_root}')
        ir_by_stem = _index_images_by_stem(ir_root)
        pairs = []
        missing = []
        for vis_path in vis_paths:
            ir_path = osp.join(ir_root, osp.basename(vis_path))
            if not osp.isfile(ir_path):
                vis_stem = osp.splitext(osp.basename(vis_path))[0]
                ir_path = ir_by_stem.get(vis_stem, ir_path)
            if osp.isfile(ir_path):
                pairs.append([vis_path, ir_path])
            else:
                missing.append(ir_path)
        if missing:
            print_log(f'Skipped {len(missing)} visible images without IR pair.',
                      logger='current', level=logging.WARNING)
            for path in missing[:5]:
                print_log(f'Missing IR pair: {path}',
                          logger='current', level=logging.WARNING)
        if not pairs:
            vis_samples = [osp.basename(path) for path in vis_paths[:5]]
            ir_samples = [
                osp.basename(path) for path in _list_images(ir_root)[:5]
            ]
            raise RuntimeError(
                'No visible/IR image pairs were found. '
                f'Visible folder: {vis_inputs}, IR folder: {ir_root}, '
                f'visible samples: {vis_samples}, IR samples: {ir_samples}')
        return pairs

    if ir_inputs is None:
        ir_path = _infer_ir_path(vis_inputs)
    elif osp.isdir(ir_inputs):
        ir_path = osp.join(ir_inputs, osp.basename(vis_inputs))
        if not osp.isfile(ir_path):
            vis_stem = osp.splitext(osp.basename(vis_inputs))[0]
            ir_path = _index_images_by_stem(ir_inputs).get(vis_stem, ir_path)
    else:
        ir_path = ir_inputs

    if not osp.isfile(vis_inputs):
        raise FileNotFoundError(f'Visible image not found: {vis_inputs}')
    if not osp.isfile(ir_path):
        raise FileNotFoundError(f'IR image not found: {ir_path}')
    return [[vis_inputs, ir_path]]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('inputs', type=str, help='Visible image file/folder.')
    parser.add_argument(
        'model',
        type=str,
        help='Config or checkpoint .pth file or model name.')
    parser.add_argument(
        '--ir-inputs',
        type=str,
        default=None,
        help='Infrared image file/folder. If omitted, replace vis with ir.')
    parser.add_argument('--weights', default=None, help='Checkpoint file')
    parser.add_argument(
        '--out-dir', type=str, default='outputs', help='Output directory.')
    parser.add_argument(
        '--texts', help='text prompt, such as "car . bus ." or "$: coco"')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--pred-score-thr', type=float, default=0.3, help='score threshold')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--no-save-vis', action='store_true')
    parser.add_argument('--no-save-pred', action='store_true')
    parser.add_argument('--print-result', action='store_true')
    parser.add_argument(
        '--palette',
        default='none',
        choices=['coco', 'voc', 'citys', 'random', 'none'])
    parser.add_argument('--custom-entities', '-c', action='store_true')
    parser.add_argument('--chunked-size', '-s', type=int, default=-1)
    parser.add_argument('--tokens-positive', '-p', type=str)

    call_args = vars(parser.parse_args())

    if call_args['no_save_vis'] and call_args['no_save_pred']:
        call_args['out_dir'] = ''

    if call_args['model'].endswith('.pth'):
        print_log('The model is a weight file, automatically assign the '
                  'model to --weights')
        call_args['weights'] = call_args['model']
        call_args['model'] = None

    if call_args['texts'] is not None and call_args['texts'].startswith('$:'):
        dataset_name = call_args['texts'][3:].strip()
        class_names = get_classes(dataset_name)
        call_args['texts'] = tuple(class_names)

    if call_args['tokens_positive'] is not None:
        call_args['tokens_positive'] = ast.literal_eval(
            call_args['tokens_positive'])

    init_args = {}
    for init_kw in ['model', 'weights', 'device', 'palette']:
        init_args[init_kw] = call_args.pop(init_kw)

    return init_args, call_args


def main():
    init_args, call_args = parse_args()
    inferencer = DualImageDetInferencer(**init_args)

    chunked_size = call_args.pop('chunked_size')
    inferencer.model.test_cfg.chunked_size = chunked_size

    vis_inputs = call_args.pop('inputs')
    ir_inputs = call_args.pop('ir_inputs')
    texts = call_args.pop('texts')
    custom_entities = call_args.pop('custom_entities')
    tokens_positive = call_args.pop('tokens_positive')

    if texts is None:
        texts = tuple(inferencer.model.dataset_meta['classes'])
        custom_entities = True

    pairs = _build_pairs(vis_inputs, ir_inputs)
    if not pairs:
        raise RuntimeError('No visible/IR image pairs were found.')

    infer_inputs = []
    for pair in pairs:
        item = dict(
            img_path=pair,
            text=copy.deepcopy(texts),
            custom_entities=custom_entities,
            tokens_positive=copy.deepcopy(tokens_positive))
        infer_inputs.append(item)

    inferencer(infer_inputs, **call_args)

    if call_args['out_dir'] != '' and not (call_args['no_save_vis']
                                           and call_args['no_save_pred']):
        print_log(f'results have been saved at {call_args["out_dir"]}')


if __name__ == '__main__':
    main()
