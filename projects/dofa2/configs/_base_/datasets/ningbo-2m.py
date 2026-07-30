data_root = (
    '/mnt/ht2-nas2/EO_test/openmmlab-archive/dat/'
    'ningbo-slices-512-dataset-7class'
)

dataset_type = 'NingBo2MSegDataset'
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromTIF'),
    dict(type='LoadSegMapFromTIF'),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.9),
    dict(type='RandomRotate', prob=0.5, degree=90),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromTIF'),
    dict(type='LoadSegMapFromTIF'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.txt',
        img_suffix='.tif',
        seg_map_suffix='.tif',
        data_prefix=dict(
            img_path='train/images',
            seg_map_path='train/masks',
        ),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='valid.txt',
        img_suffix='.tif',
        seg_map_suffix='.tif',
        data_prefix=dict(img_path='val/images', seg_map_path='val/masks'),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
