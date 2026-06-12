_base_ = ['./upernet_copernicus-fm-base_4xb2-40k_cloud-s3-224x224.py']

crop_size = (224, 224)
ignore_index = 255
patch_area = (8 * 300 / 1000)**2
s3_band_scales = [
    0.0139465, 0.0133873, 0.0121481, 0.0115198, 0.0100953, 0.0123538,
    0.00879161, 0.00876539, 0.0095103, 0.00773378, 0.00675523, 0.0071996,
    0.00749684, 0.0086512, 0.00526779, 0.00530267, 0.00493004, 0.00549962,
    0.00502847, 0.00326378, 0.00324118
]
cloud_s3_binary_label_mapping = {0: 255, 1: 0, 2: 1}

train_pipeline = [
    dict(
        type='LoadCopernicusGeoTiffImageFromFile',
        band_scales=s3_band_scales,
        date_separator='____',
        date_token_index=1,
        patch_area=patch_area),
    dict(
        type='LoadCoBenchSegAnnotations',
        label_mapping=cloud_s3_binary_label_mapping,
        default_value=ignore_index),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(
        type='RandomRotate',
        prob=0.5,
        degree=90,
        pad_val=0,
        seg_pad_val=ignore_index),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='PackSegInputs',
        meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                   'reduce_zero_label', 'copernicus_meta')),
]
test_pipeline = [
    dict(
        type='LoadCopernicusGeoTiffImageFromFile',
        band_scales=s3_band_scales,
        date_separator='____',
        date_token_index=1,
        patch_area=patch_area),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(
        type='LoadCoBenchSegAnnotations',
        label_mapping=cloud_s3_binary_label_mapping,
        default_value=ignore_index),
    dict(
        type='PackSegInputs',
        meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                   'reduce_zero_label', 'copernicus_meta')),
]

model = dict(
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2),
)

train_dataloader = dict(
    dataset=dict(
        data_prefix=dict(img_path='s3_olci', seg_map_path='cloud_binary'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    dataset=dict(
        data_prefix=dict(img_path='s3_olci', seg_map_path='cloud_binary'),
        pipeline=test_pipeline,
    ))
test_dataloader = dict(
    dataset=dict(
        data_prefix=dict(img_path='s3_olci', seg_map_path='cloud_binary'),
        pipeline=test_pipeline,
    ))
