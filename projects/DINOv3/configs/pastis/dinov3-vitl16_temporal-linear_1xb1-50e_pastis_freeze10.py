# First 10 epochs: frozen DINOv3. Epochs 11-50: full fine-tuning.
BACKBONE_INITIAL_FREEZE = False
OPTIM_WRAPPER = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'norm': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
        }
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
CUSTOM_HOOKS = [dict(type='BackboneFreezeSwitchHook', freeze_epochs=10)]
# DDP is built while all parameters require gradients; the hook then freezes
# the backbone. This flag prevents reduction hangs during the frozen phase.
FIND_UNUSED_PARAMETERS = True


import os

custom_imports = dict(
    imports=['projects.DINOv3'],
    allow_failed_imports=False,
)

# ---- User-editable paths/interfaces ----
data_root = os.getenv('PASTIS_DATA_ROOT', '/path/to/pastis_dataset_64')
dinov3_weights_path = os.getenv(
    'DINOV3_SAT493M_WEIGHTS',
    '/path/to/dinov3_vitl16_pretrain_sat493m.pth',
)
norm_stats_path = os.getenv(
    'PASTIS_NORM_STATS',
    'projects/DINOv3/configs/pastis/NORM_S2_patch.json',
)

# Change to None to keep 64x64, or use any H/W divisible by 16.
resize_size = (224, 224)
# Supported values: 'mean' and 'max'.
temporal_fusion = 'mean'
# The same train-fold statistics are used for train/val/test.
norm_folds = ('Fold_1', 'Fold_2', 'Fold_3')
channel_map_10_to_13 = (0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 8, 9)

num_classes = 19
ignore_index = 255
patch_size = 16
hidden_dim = 1024
max_epochs = 50

train_pipeline = [
    dict(
        type='LoadPastisSampleFromPT',
        expected_times=12,
        expected_channels=(10, 13),
        source_ignore_index=-1,
        target_ignore_index=ignore_index,
    ),
    dict(type='PastisResize', size=resize_size),
    dict(
        type='NormalizePastisFromJSON',
        stats_file=norm_stats_path,
        folds=norm_folds,
        channel_map_10_to_13=channel_map_10_to_13,
        adapt_image_10_to_13=True,
    ),
    dict(type='PastisPackSegInputs'),
]
val_pipeline = train_pipeline

train_dataloader = dict(
    batch_size=1,  # per GPU
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='PastisPTDataset',
        data_root=data_root,
        split='pastis_r_train',
        pipeline=train_pipeline,
        ignore_index=ignore_index,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='PastisPTDataset',
        data_root=data_root,
        split='pastis_r_val',
        pipeline=val_pipeline,
        ignore_index=ignore_index,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='PastisPTDataset',
        data_root=data_root,
        split='pastis_r_test',
        pipeline=val_pipeline,
        ignore_index=ignore_index,
    ),
)

val_evaluator = dict(
    type='IoUMetric',
    ignore_index=ignore_index,
    iou_metrics=['mIoU', 'mDice'],
)
test_evaluator = val_evaluator

model = dict(
    type='TemporalEncoderDecoder',
    data_preprocessor=dict(
        type='TemporalSegDataPreProcessor',
        pad_val=0.0,
        seg_pad_val=ignore_index,
        size_divisor=patch_size,
    ),
    input_projection=dict(
        type='SpectralProjection',
        in_channels=13,
        out_channels=3,
        with_norm=True,
    ),
    temporal_fusion=temporal_fusion,
    backbone=dict(
        type='DINOv3ViT',
        model_name='dinov3_vitl16',
        weights_name='SAT493M',
        weights_path=dinov3_weights_path,
        out_indices=(23,),
        patch_size=patch_size,
        load_strict=True,
        freeze=BACKBONE_INITIAL_FREEZE,
        norm=True,
    ),
    decode_head=dict(
        type='LinearProbeHead',
        in_channels=hidden_dim,
        in_index=0,
        num_classes=num_classes,
        dropout_ratio=0.0,
        align_corners=False,
        ignore_index=ignore_index,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, begin=0, end=1, by_epoch=True),
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-6,
        begin=1,
        end=max_epochs,
        T_max=max_epochs - 1,
        by_epoch=True,
    ),
]

optim_wrapper = OPTIM_WRAPPER

custom_hooks = CUSTOM_HOOKS

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=5,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=3,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=False, interval=1),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)

# Standard MMEngine settings work with tools/train.py (single GPU) and
# tools/dist_train.sh (multi GPU).
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
find_unused_parameters = FIND_UNUSED_PARAMETERS

default_scope = 'mmseg'
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=0, deterministic=False)
auto_scale_lr = dict(enable=False, base_batch_size=1)
