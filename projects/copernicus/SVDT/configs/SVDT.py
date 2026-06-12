
import os
dataset_type = 'LocalSVDTDataset'
data_root = os.path.join(
    os.environ.get('MM_ARCHIVE_DATA_HOME'), 
    'SVDT',
)
crop_size = (256, 256)
patch_area = (16 * 10 / 1000)**2


s2_band_stats = dict(
    mean=[
        69.6123, 89.7399, 72.4085,
    ],
    std=[
        23.1234, 23.9954, 32.8544,
    ],
)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
)

train_pipeline = [
    dict(type='LoadSinglePNGImageFromFile'),
    dict(type='LoadLocalSVDTAnnotations'),
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
    dict(type='LoadSinglePNGImageFromFile'),
    dict(type='AddCopernicusMeta', patch_area=patch_area),
    dict(
        type='NormalizeMultibandImage',
        mean=s2_band_stats['mean'],
        std=s2_band_stats['std']),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='LoadLocalSVDTAnnotations'),
    dict(
        type='PackSegInputs',
        meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                   'reduce_zero_label', 'copernicus_meta')),
]

train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir', seg_map_path='ann_dir'),
        ann_file='train.txt',
        ignore_index=255,
        reduce_zero_label=False,
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir', seg_map_path='ann_dir'),
        ann_file='valid.txt',
        ignore_index=255,
        reduce_zero_label=False,
        pipeline=test_pipeline,
    ))
test_dataloader = val_dataloader