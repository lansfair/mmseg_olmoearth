import os


_base_ = [
    '../../../../configs/_base_/default_runtime.py',
    './cashew-plant.py'
]

custom_imports = dict(
    imports=['projects.copernicus.m-cashew-plant.copernicus'], 
    allow_failed_imports=False
)

PATCH_AREA = (16 * 10 / 1000)**2

S2_BAND_WAVELENGTHS = (
    440,  # 01 - Coastal aerosol
    490,  # 02 - Blue
    560,  # 03 - Green
    665,  # 04 - Red
    705,  # 05 - Vegetation Red Edge
    740,  # 06 - Vegetation Red Edge
    783,  # 07 - Vegetation Red Edge
    842,  # 08 - NIR
    860,  # 08A - Vegetation Red Edge
    940,  # 09 - Water vapour
    1370, # 11 - SWIR
    1610, # 12 - SWIR
    2190  # Cloud Probability
)

S2_BAND_BANDWIDTHS = (
    20,  # 01 - Coastal aerosol
    65,  # 02 - Blue
    35,  # 03 - Green
    30,  # 04 - Red
    15,  # 05 - Vegetation Red Edge
    15,  # 06 - Vegetation Red Edge
    20,  # 07 - Vegetation Red Edge
    115, # 08 - NIR
    20,  # 08A - Vegetation Red Edge
    20,  # 09 - Water vapour
    30,  # 11 - SWIR
    90,  # 12 - SWIR
    180  # Cloud Probability
)

BACKBONE_EMBED_DIM = 768
BACKBONE_CHECKPOINT_PATH = os.path.join(os.environ.get('MM_ARCHIVE_CKPT_HOME'), "CopernicusFM_ViT_base_varlang_e100.pth")

IMAGE_SIZE = 256

NUM_CLASSES = 7

TRAIN_EPOCH = 40

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='CopernicusEncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        size=(IMAGE_SIZE, IMAGE_SIZE),
        bgr_to_rgb=False,
        pad_val=0,
        seg_pad_val=0
    ),
    backbone=dict(
        type='CopernicusFMBackbone',
        arch='base',
        frozen_exclude=[],
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint=BACKBONE_CHECKPOINT_PATH),
        band_wavelengths=S2_BAND_WAVELENGTHS,
        band_bandwidths=S2_BAND_BANDWIDTHS,
        var_option='spectrum',
        input_mode='spectral',
        kernel_size=16,
        patch_area=PATCH_AREA,
    ),
    decode_head=dict(
        type='LPHead',
        in_channels=BACKBONE_EMBED_DIM,
        channels=BACKBONE_EMBED_DIM,
        num_classes=NUM_CLASSES,
        dropout_ratio=0
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=TRAIN_EPOCH, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [ 
    dict(
        type='OneCycleLR',
        eta_max=1e-3,
        pct_start=0.0,
        anneal_strategy='cos',
        begin=0,
        end=TRAIN_EPOCH,
        by_epoch=True,
        convert_to_iter_based=True,
    )
]

log_processor = dict(by_epoch=True)
