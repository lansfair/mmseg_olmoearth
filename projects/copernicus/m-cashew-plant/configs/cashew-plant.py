import os


DATA_ROOT = os.path.join(
    os.environ.get('MM_ARCHIVE_DATA_HOME'), 
    'geo-bench-1.0',
    'segmentation_v1.0',
    'm-cashew-plant'
)

DATASET_TYPE = 'CashewPlantSegDataset'

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
    520.1185302734375,   # 01 - Coastal aerosol
    634.7583618164062,   # 02 - Blue
    892.4611816406250,   # 03 - Green
    880.7075805664062,   # 04 - Red
    1380.6409912109375,  # 05 - Vegetation Red Edge
    2233.432373046875,   # 06 - Vegetation Red Edge
    2549.379638671875,   # 07 - Vegetation Red Edge
    2643.248046875,      # 08 - NIR
    2643.531982421875,   # 08A - Vegetation Red Edge
    2852.87451171875,    # 09 - Water vapour
    2463.933349609375,   # 11 - SWIR
    1600.9207763671875,  # 12 - SWIR
    0.010281000286340714 # Cloud Probability
]

BANDS_STD = [
    204.20234680175780,  # 01 - Coastal aerosol
    227.25344848632812,  # 02 - Blue
    222.32545471191406,  # 03 - Green
    350.47235107421875,  # 04 - Red
    280.64367675781250,  # 05 - Vegetation Red Edge
    373.75210571289060,  # 06 - Vegetation Red Edge
    449.92361450195310,  # 07 - Vegetation Red Edge
    414.64981079101560,  # 08 - NIR
    415.10195922851560,  # 08A - Vegetation Red Edge
    413.89804077148440,  # 09 - Water vapour
    494.97430419921875,  # 11 - SWIR
    514.42297363281250,  # 12 - SWIR
    0.3447800576686859   # Cloud Probability
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

U16_BAND_DATA = True

TRAIN_BATCH_SIZE = 32


train_pipeline = [
    dict(type='LoadSingleRSImgFromHDF5', bands=BANDS_NAME, convert_to_u16=U16_BAND_DATA),
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
    batch_size=TRAIN_BATCH_SIZE,
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
    batch_size=1,
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
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False)
)
