_base_ = ['./dofav2-large_4xb16-20e_m-cashew-s2-p16-paper.py']

# Diagnostic configuration matching the successful RGB experiment's 256x256
# geometry and 30-epoch schedule while retaining all nine Sentinel-2 bands.
crop_size = (256, 256)

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
    backbone=dict(img_size=256),
)
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

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
