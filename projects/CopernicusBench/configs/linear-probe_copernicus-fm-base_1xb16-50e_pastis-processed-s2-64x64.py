_base_ = ['../../../configs/_base_/default_runtime.py']

custom_imports = dict(
    imports=['projects.CopernicusBench.copernicusbench'],
    allow_failed_imports=False)

dataset_type = 'PASTISProcessedS2Dataset'
data_root = 'data/pastis_r'
crop_size = (64, 64)
ignore_index = 255
patch_area = (16 * 10 / 1000)**2
copernicus_fm_checkpoint = (
    'https://huggingface.co/wangyi111/Copernicus-FM/resolve/main/'
    'CopernicusFM_ViT_base_varlang_e100.pth')

# OlmoEarth processed PASTIS saves 13 eval bands:
# B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12.
# Keep all bands to match the existing CopernicusBench Sentinel-2 configs.
pastis_s2_band_indices = list(range(13))
s2_band_wavelengths = [
    440, 490, 560, 665, 705, 740, 783, 842, 860, 940, 1370, 1610, 2190
]
s2_band_bandwidths = [20, 65, 35, 30, 15, 15, 20, 115, 20, 20, 30, 90, 180]

copernicus_s2_stats = dict(
    mean=[
        1353.7, 1117.2, 1041.8, 946.5, 1199.1, 2003.0, 2374.0, 2301.2,
        2599.7, 732.1, 12.1, 1820.6, 1118.2
    ],
    std=[
        897.3, 736.0, 684.8, 620.0, 791.9, 1341.3, 1595.4, 1545.5,
        1750.1, 475.1, 98.3, 1216.5, 736.7
    ],
)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=ignore_index,
)

train_pipeline = [
    dict(
        type='LoadPastisProcessedS2TimeSeriesFromFile',
        band_indices=pastis_s2_band_indices,
        patch_area=patch_area),
    dict(type='LoadPastisProcessedAnnotations', ignore_index=ignore_index),
    dict(
        type='NormalizeMultibandImage',
        mean=copernicus_s2_stats['mean'],
        std=copernicus_s2_stats['std']),
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
        type='LoadPastisProcessedS2TimeSeriesFromFile',
        band_indices=pastis_s2_band_indices,
        patch_area=patch_area),
    dict(
        type='NormalizeMultibandImage',
        mean=copernicus_s2_stats['mean'],
        std=copernicus_s2_stats['std']),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='LoadPastisProcessedAnnotations', ignore_index=ignore_index),
    dict(
        type='PackSegInputs',
        meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                   'reduce_zero_label', 'copernicus_meta')),
]

model = dict(
    type='TemporalCopernicusEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='CopernicusFMBackbone',
        arch='base',
        frozen_exclude=[],
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint=copernicus_fm_checkpoint),
        band_wavelengths=s2_band_wavelengths,
        band_bandwidths=s2_band_bandwidths,
        var_option='spectrum',
        input_mode='spectral',
        kernel_size=16,
        patch_area=patch_area,
    ),
    decode_head=dict(
        type='PatchLinearHead',
        in_channels=768,
        in_index=3,
        channels=768,
        dropout_ratio=0.0,
        num_classes=19,
        output_patch_size=16,
        align_corners=False,
        ignore_index=ignore_index,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            avg_non_ignore=True,
            loss_weight=1.0)),
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
        split='train',
        ignore_index=ignore_index,
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
        split='valid',
        ignore_index=ignore_index,
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
        split='test',
        ignore_index=ignore_index,
        reduce_zero_label=False,
        pipeline=test_pipeline,
    ))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
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
        save_last=True))
