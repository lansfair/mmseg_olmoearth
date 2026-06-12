import os
_base_ = [
    '../../../../configs/_base_/default_runtime.py',
    './SVDT.py'
]

custom_imports = dict(
    imports=['projects.copernicus.SVDT.copernicus'],
    allow_failed_imports=False)

crop_size = (256, 256)
patch_area = (16 * 10 / 1000)**2


copernicus_fm_checkpoint = os.path.join(os.environ.get('MM_ARCHIVE_CKPT_HOME'), "CopernicusFM_ViT_base_varlang_e100.pth")
s2_band_wavelengths = [
    665, 560, 490,
]
s2_band_bandwidths = [30, 35, 65]

norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
)


model = dict(
    type='CopernicusEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='CopernicusFMBackbone',
        arch='base',
        frozen_exclude=[],
        norm_eval=False,
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
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

val_evaluator = dict(type='IoUMetric', iou_metrics=["mIoU", "mFscore"])
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
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

log_processor = dict(by_epoch=True)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))