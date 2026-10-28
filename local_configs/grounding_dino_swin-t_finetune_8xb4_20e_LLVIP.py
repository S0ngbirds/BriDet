_base_ = 'grounding_dino_swin-t_pretrain_obj365.py'

data_root = '/root/autodl-tmp/mmdetection/data/LLVIP'
backend_args = None
dataset_type = 'PairedCocoDataset'


model = dict(
    bbox_head=dict(
        num_classes=1,),
    fusion_module = dict(
        use_fusion=True,
        type = "add",
        position="after_backbone"),
    use_sr_branch=False,
    sr_branch_cfg=dict(
        type='SRBranch',
        in_channels=512,          
        out_channels=6,           
        factor=3,                 
        num_layers=3,            
        intermediate_channels=64, 
    ),
    # llm_cfg = dict(
    #     lmm_region_loss_weight=1.0,
    #     lmm_image_loss_weight=1.0,
    #     max_new_tokens = 300,
    #     temperature = 0.7,
    #     do_sample = True,
    # ),
    data_preprocessor=dict(
        type='DetMultiChannelDataPreprocessor',
        mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=False,))


train_pipeline = [
    dict(
        type='LoadMultiChannelImageFromFiles',
        color_type='color',
        backend_args=None,
        imdecode_backend='cv2'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))]

DV_Dataset = dict(
    _delete_=True,
    metainfo = dict(
        classes= ('person',),
        palette=[(220, 20, 60),]
    ),
    type='PairedCocoDataset',
    data_root=data_root,
    return_classes=True,
    pipeline=train_pipeline,
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    ann_file=data_root + '/annotations/train.json',
    data_prefix=dict(
            imgc=data_root + '/vis/train/',
            imgd=data_root + '/ir/train/'))


dataset = DV_Dataset

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset = DV_Dataset,
)

test_pipeline = [
    dict(
        type='LoadMultiChannelImageFromFiles',
        color_type='color',
        backend_args=None,
        imdecode_backend='cv2'),
    dict(
        type='SpiltMultiChannel',
        channels=[3, 3],
        transforms=[
            dict(
                type='FixScaleResize',
                scale=(512, 640),
                # scale=(712, 840),
                keep_ratio=True,
                backend='pillow'),
            ]),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))]

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type='PairedCocoDataset',
        pipeline=test_pipeline,
        data_root=data_root,
        ann_file=data_root + '/annotations/test.json',
        data_prefix=dict(
            _delete_=True,
            imgc=data_root + '/vis/test',
            imgd=data_root + '/ir/test')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root+'/annotations/test.json',
                    classwise=True,)
test_evaluator = val_evaluator

max_epoch = 20

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best=['coco/bbox_mAP_50']),
    logger=dict(type='LoggerHook', interval=10))
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[15],
        gamma=0.1)
]
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True
)
optim_wrapper = dict(
    optimizer=dict(lr=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0.0),
            # 'encoder': dict(lr_mult=0.1),
            # 'decoder': dict(lr_mult=0.0),
            # 'bbox_head': dict(lr_mult=0.0),
            'sr_branch': dict(lr_mult=0.1),
        }))
# find_unused_parameters = True
# detect_anomalous_params = True
# find_unused_parameters=False

# load_from = "/root/autodl-tmp/CrlDet/drone_single/best_coco_bbox_mAP_50_epoch_5.pth"
load_from = "/root/autodl-tmp/mmdetection/weights/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth"