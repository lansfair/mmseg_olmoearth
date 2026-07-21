_base_ = ['../../../configs/_base_/default_runtime.py']

custom_imports = dict(
    imports=['projects.CopernicusBench.copernicusbench'],
    allow_failed_imports=False)

dataset_type = 'CopernicusBenchDFC2020S2Dataset'
data_root = 'data/copernicusbench/dfc2020_s1s2'
crop_size = (256, 256)
patch_area = (16 * 10 / 1000)**2
copernicus_fm_checkpoint = (
    'https://huggingface.co/wangyi111/Copernicus-FM/resolve/main/'
    'CopernicusFM_ViT_base_varlang_e100.pth')

classes = (
    'Forest',
    'Shrubland',
    'Grassland',
    'Wetland',
    'Cropland',
    'Urban/Built-up',
    'Barren',
    'Water',
)
palette = [
    [0, 100, 0],
    [255, 187, 34],
    [255, 255, 76],
    [0, 150, 160],
    [240, 150, 255],
    [250, 0, 0],
    [180, 180, 180],
    [0, 100, 200],
]
visualizer = dict(classes=classes, palette=palette)

s2_band_wavelengths = [
    440, 490, 560, 665, 705, 740, 783, 842, 860, 940, 1370, 1610, 2190
]
s2_band_bandwidths = [20, 65, 35, 30, 15, 15, 20, 115, 20, 20, 30, 90, 180]
s2_band_stats = dict(
    mean=[
        1353.7, 1117.2, 1041.8, 946.5, 1199.1, 2003.0, 2374.0, 2301.2,
        2599.7, 732.1, 12.1, 1820.6, 1118.2
    ],
    std=[
        897.3, 736.0, 684.8, 620.0, 791.9, 1341.3, 1595.4, 1545.5,
        1750.1, 475.1, 98.3, 1216.5, 736.7
    ],
)

norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
)

train_pipeline = [
    dict(type='LoadSingleRSImageFromFile'),
    dict(type='LoadDFC2020Annotations'),
    dict(type='AddCopernicusMeta', patch_area=patch_area),
    dict(
        type='NormalizeMultibandImage',
        mean=s2_band_stats['mean'],
        std=s2_band_stats['std']),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='RandomRotate', prob=0.5, degree=90, pad_val=0, seg_pad_val=255),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='PackSegInputs',
        meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                   'reduce_zero_label', 'copernicus_meta')),
]
test_pipeline = [
    dict(type='LoadSingleRSImageFromFile'),
    dict(type='AddCopernicusMeta', patch_area=patch_area),
    dict(
        type='NormalizeMultibandImage',
        mean=s2_band_stats['mean'],
        std=s2_band_stats['std']),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='LoadDFC2020Annotations'),
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
        init_cfg=dict(
            type='Pretrained',
            checkpoint=copernicus_fm_checkpoint,
        ),
        band_wavelengths=s2_band_wavelengths,
        band_bandwidths=s2_band_bandwidths,
        var_option='spectrum',
        input_mode='spectral',
        kernel_size=16,
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
        num_classes=8,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            avg_non_ignore=True,
            loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=8,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            avg_non_ignore=True,
            loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    pin_memory=True,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='s2', seg_map_path='dfc'),
        ann_file='dfc-train-new.csv',
        ignore_index=255,
        reduce_zero_label=False,
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='s2', seg_map_path='dfc'),
        ann_file='dfc-val-new.csv',
        ignore_index=255,
        reduce_zero_label=False,
        pipeline=test_pipeline,
    ))
test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='s2', seg_map_path='dfc'),
        ann_file='dfc-test-new.csv',
        ignore_index=255,
        reduce_zero_label=False,
        pipeline=test_pipeline,
    ))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
log_processor = dict(by_epoch=True)
randomness = dict(seed=0)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=0.01),
)
param_scheduler = [
    dict(
        type='OneCycleLR',
        eta_max=1e-3,
        pct_start=0.0,
        anneal_strategy='cos',
        begin=0,
        end=50,
        by_epoch=True,
        convert_to_iter_based=True,
    )
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=1,
        max_keep_ckpts=1,
        save_best='mIoU',
        rule='greater',
        save_last=True),
    visualization=dict(
        type='CopernicusSegVisualizationHook',
        draw=True,
        interval=50,
        rgb_band_indices=(3, 2, 1)))
