custom_imports = dict(imports=['projects.dofa2.dofa2'])

checkpoint = (
    '/mnt/ht2-nas2/EO_test/openmmlab-archive/pretrained/'
    'dofav2_vit_large_e150.pth'
)

norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=None,
    std=None,
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
    # The dataset pipelines already crop samples to their configured size.
    # MMSeg still requires exactly one of size/size_divisor for batch stacking;
    # divisor 1 preserves those crop sizes without introducing extra padding.
    size_divisor=1,
)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='DOFAV2ViT',
        arch='large',
        img_size=224,
        patch_size=14,
        model_bands=[
            'RED',
            'GREEN',
            'BLUE',
            'RED_EDGE_1',
            'RED_EDGE_2',
            'RED_EDGE_3',
            'NIR_BROAD',
            'SWIR_1',
            'SWIR_2',
        ],
        out_indices=(5, 11, 17, 23),
        convert_patch_14_to_16=False,
        drop_path_rate=0.0,
        freeze_backbone=True,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
    ),
    # Match the paper/official implementation: turn the four ViT feature
    # levels into a pyramid with learned transposed convolutions and pooling.
    neck=dict(
        type='Feature2Pyramid',
        embed_dim=1024,
        rescales=[4, 2, 1, 0.5],
        norm_cfg=norm_cfg,
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[1024, 1024, 1024, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=norm_cfg,
        align_corners=False,
        ignore_index=255,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
        ),
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=norm_cfg,
        align_corners=False,
        ignore_index=255,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4,
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)
