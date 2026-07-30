dataset_root = (
    '/mnt/ht2-nas2/EO_test/openmmlab-archive/dat/geo-bench-1.0/'
    'segmentation_v1.0/m-cashew-plant'
)

dataset_type = 'CashewPlantSegDataset'
crop_size = (224, 224)

train_pipeline = [
    dict(type='LoadDOFAGeoBenchSample', num_classes=7),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.9),
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

train_dataloader = dict(
    batch_size=12,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        dataset_root=dataset_root,
        split='train',
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=12,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        dataset_root=dataset_root,
        split='valid',
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        dataset_root=dataset_root,
        split='test',
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
