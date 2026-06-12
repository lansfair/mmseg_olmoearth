_base_ = ['../../../../configs/_base_/default_runtime.py']

custom_imports = dict(
    imports=['projects.CopernicusBench.copernicusbench'],
    allow_failed_imports=False)

dataset_type = 'CloudS3Dataset'
data_root = 'data/copernicusbench/cloud_s3'
crop_size = (224, 224)
ignore_index = 255
patch_area = (8 * 300 / 1000)**2
copernicus_fm_checkpoint = (
    'https://huggingface.co/wangyi111/Copernicus-FM/resolve/main/'
    'CopernicusFM_ViT_base_varlang_e100.pth')

s3_band_wavelengths = [
    400, 412.5, 442.5, 490, 510, 560, 620, 665, 673.75, 681.25, 708.75,
    753.75, 761.25, 764.375, 767.5, 778.75, 865, 885, 900, 940, 1020
]
s3_band_bandwidths = [
    15, 10, 10, 10, 10, 10, 10, 10, 7.5, 7.5, 10, 7.5, 7.5, 3.75, 2.5, 15,
    20, 10, 10, 20, 40
]
s3_band_scales = [
    0.0139465, 0.0133873, 0.0121481, 0.0115198, 0.0100953, 0.0123538,
    0.00879161, 0.00876539, 0.0095103, 0.00773378, 0.00675523, 0.0071996,
    0.00749684, 0.0086512, 0.00526779, 0.00530267, 0.00493004, 0.00549962,
    0.00502847, 0.00326378, 0.00324118
]
cloud_s3_label_mapping = {0: 255, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=ignore_index,
)

train_pipeline = [
    dict(
        type='LoadCopernicusGeoTiffImageFromFile',
        band_scales=s3_band_scales,
        date_separator='____',
        date_token_index=1,
        patch_area=patch_area),
    dict(
        type='LoadCoBenchSegAnnotations',
        label_mapping=cloud_s3_label_mapping,
        default_value=ignore_index),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(
        type='RandomRotate',
        prob=0.5,
        degree=90,
        pad_val=0,
        seg_pad_val=ignore_index),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='PackSegInputs',
        meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                   'reduce_zero_label', 'copernicus_meta')),
]
test_pipeline = [
    dict(
        type='LoadCopernicusGeoTiffImageFromFile',
        band_scales=s3_band_scales,
        date_separator='____',
        date_token_index=1,
        patch_area=patch_area),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(
        type='LoadCoBenchSegAnnotations',
        label_mapping=cloud_s3_label_mapping,
        default_value=ignore_index),
    dict(
        type='PackSegInputs',
        meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                   'reduce_zero_label', 'copernicus_meta')),
]

model = dict(
    type='CopernicusEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='CopernicusFMBackbone',
        arch='base',
        frozen_exclude=[],
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint=copernicus_fm_checkpoint),
        band_wavelengths=s3_band_wavelengths,
        band_bandwidths=s3_band_bandwidths,
        var_option='spectrum',
        input_mode='spectral',
        kernel_size=8,
        patch_area=patch_area,
    ),
    neck=dict(type='Feature2Pyramid', embed_dim=768, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type='UPerHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            ignore_index=ignore_index)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4,
            ignore_index=ignore_index)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='s3_olci', seg_map_path='cloud_multi'),
        ann_file='train.csv',
        ignore_index=ignore_index,
        reduce_zero_label=False,
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='s3_olci', seg_map_path='cloud_multi'),
        ann_file='val.csv',
        ignore_index=ignore_index,
        reduce_zero_label=False,
        pipeline=test_pipeline,
    ))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='s3_olci', seg_map_path='cloud_multi'),
        ann_file='test.csv',
        ignore_index=ignore_index,
        reduce_zero_label=False,
        pipeline=test_pipeline,
    ))

val_evaluator = dict(
    type='IoUMetric', iou_metrics=['mIoU'], ignore_index=ignore_index)
test_evaluator = val_evaluator

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=40000,
        by_epoch=False,
    )
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000))
