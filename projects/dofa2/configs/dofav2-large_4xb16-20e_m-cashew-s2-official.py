_base_ = ['./dofav2-large_1xb12-20e_m-cashew-s2-frozen.py']

# DOFA+ ViT-L on GEO-Bench m-cashew-plant.
#
# The public paper specifies a global batch size of 64. The official
# downstream code later published a task-specific learning rate of 1e-3 for
# m-cashew, plus three warm-up epochs and a minimum learning rate of 1e-6.
# Four GPUs x sixteen samples preserves the paper's global batch size.
crop_size = (224, 224)

train_pipeline = [
    dict(type='LoadDOFAGeoBenchSample', num_classes=7),
    dict(type='CenterCrop', crop_size=crop_size),
    dict(type='RandomRotate', prob=0.5, degree=90),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=16,
    dataset=dict(pipeline=train_pipeline),
)
val_dataloader = dict(batch_size=16)
test_dataloader = dict(batch_size=16)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=0.01),
)
param_scheduler = [
    dict(
        type='LinearLR',
        by_epoch=True,
        begin=0,
        end=3,
        start_factor=1e-3,
    ),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=3,
        end=20,
        T_max=17,
        eta_min=1e-6,
    ),
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=20,
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
randomness = dict(seed=42, deterministic=False)
