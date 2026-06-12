import os


DATA_ROOT = os.path.join(
    os.environ.get('MM_ARCHIVE_DATA_HOME'), 
    'geo-bench-1.0',
    'segmentation_v1.0',
    'm-sa-crop-type'
)

DATASET_TYPE = 'SACropTypeSegDataset'

BANDS_NAME = [
    '01 - Coastal aerosol',
    '02 - Blue',
    '03 - Green',
    '04 - Red',
    '05 - Vegetation Red Edge',
    '06 - Vegetation Red Edge',
    '07 - Vegetation Red Edge',
    '08 - NIR',
    '08A - Vegetation Red Edge',
    '09 - Water vapour',
    '11 - SWIR',
    '12 - SWIR',
    'Cloud Probability'
]

BANDS_MEAN = [
    12.739611,
    16.526744,
    26.636417,
    36.696639,
    46.388679,
    58.281453,
    63.575819,
    68.1836,
    69.142591,
    69.904566,
    83.626811,
    65.767679,
    0.0
]

BANDS_STD = [
    7.492811526301659,
    9.329547939662671,
    12.674537246073758,
    19.421922023931593,
    19.487411106531287,
    19.959174612412983,
    21.53805760692545,
    23.05077775347288,
    22.329695761624677,
    21.877766438821954,
    28.14418826277069,
    27.2346215312965,
    1.0
]

PATCH_AREA = (16 * 10 / 1000)**2

META_KEYS = (
    'img_path', 
    'seg_map_path', 
    'ori_shape', 
    'img_shape',
    'pad_shape', 
    'scale_factor', 
    'flip', 
    'flip_direction',
    'reduce_zero_label', 
    'copernicus_meta'
)


train_pipeline = [
    dict(type='LoadSingleRSImgFromHDF5', bands=BANDS_NAME),
    dict(type='LoadSingleRSAnnFromHDF5'),
    dict(type='AddCopernicusMeta', patch_area=PATCH_AREA),
    dict(type='NormalizeMultibandImage', mean=BANDS_MEAN, std=BANDS_STD),
    dict(type='RandomRotate', prob=0.5, degree=90, pad_val=0, seg_pad_val=0),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PackSegInputs', meta_keys=META_KEYS)
]
train_dataloader = dict(
    dataset=dict(
        type=DATASET_TYPE,
        data_root=DATA_ROOT,
        split='train',
        pipeline=train_pipeline,
        reduce_zero_label=False
    ),
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True)
)


test_pipeline = [
    dict(type='LoadSingleRSImgFromHDF5', bands=BANDS_NAME),
    dict(type='AddCopernicusMeta', patch_area=PATCH_AREA),
    dict(type='NormalizeMultibandImage', mean=BANDS_MEAN, std=BANDS_STD),
    dict(type='LoadSingleRSAnnFromHDF5'),
    dict(type='PackSegInputs', meta_keys=META_KEYS)
]
val_dataloader = dict(
    dataset=dict(
        type=DATASET_TYPE,
        data_root=DATA_ROOT,
        split='valid',
        test_mode=True,
        pipeline=test_pipeline,
        reduce_zero_label=False
    ),
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False)
)

test_dataloader = dict(
    dataset=dict(
        type=DATASET_TYPE,
        data_root=DATA_ROOT,
        split='test',
        test_mode=True,
        pipeline=test_pipeline,
        reduce_zero_label=False
    ),
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False)
)
