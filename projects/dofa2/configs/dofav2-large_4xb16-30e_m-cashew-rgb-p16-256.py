_base_ = [
    './_base_/models/dofav2_upernet.py',
    './_base_/datasets/m-cashew-plant.py',
    './_base_/schedules/20e.py',
    '../../../configs/_base_/default_runtime.py',
]

# Best verified m-cashew-plant recipe: RGB, 256x256 input, an effective
# 16x16 patch kernel, frozen DOFAv2 ViT-L, and a global batch size of 64.
crop_size = (256, 256)
rgb_band_names = (
    '04 - Red',
    '03 - Green',
    '02 - Blue',
)

train_pipeline = [
    dict(type='LoadDOFAGeoBenchSample', num_classes=7),
    dict(type='CenterCrop', crop_size=crop_size),
    dict(type='RandomRotate', prob=0.5, degree=90),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadDOFAGeoBenchSample', num_classes=7),
    dict(type='CenterCrop', crop_size=crop_size),
    dict(type='PackSegInputs'),
]

model = dict(
    backbone=dict(
        img_size=256,
        model_bands=['RED', 'GREEN', 'BLUE'],
        convert_patch_14_to_16=True,
    ),
)

train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        band_names=rgb_band_names,
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        band_names=rgb_band_names,
        pipeline=test_pipeline,
    ),
)
test_dataloader = dict(
    batch_size=16,
    dataset=dict(
        band_names=rgb_band_names,
        pipeline=test_pipeline,
    ),
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-3, weight_decay=0.01),
)
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=0,
        end=30,
        T_max=30,
        eta_min=0.0,
    ),
]
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=30,
    val_interval=1,
)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=1,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=3,
    ),
)

auto_scale_lr = dict(enable=False, base_batch_size=64)
log_processor = dict(by_epoch=True)
randomness = dict(seed=42, deterministic=False)
