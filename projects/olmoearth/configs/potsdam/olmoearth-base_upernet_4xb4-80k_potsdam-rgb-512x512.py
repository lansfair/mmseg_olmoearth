_base_ = "./olmoearth-base_4xb4-50e_potsdam-rgb.py"

work_dir = (
    "./work_dirs/olmoearth-base_upernet_4xb4-80k_potsdam-rgb-512x512"
)

crop_size = 512
patch_size = 4
num_classes = 6
ignore_index = 255
num_timesteps = 1
norm_cfg = dict(type="SyncBN", requires_grad=True)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(
        type="RandomResize",
        scale=(crop_size, crop_size),
        ratio_range=(0.5, 2.0),
        keep_ratio=True,
    ),
    dict(
        type="RandomCrop",
        crop_size=(crop_size, crop_size),
        cat_max_ratio=0.75,
    ),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(
        type="RGBToOlmoEarthS2",
        num_timesteps=num_timesteps,
        rgb_channel_order="BGR",
        input_value_range="0_255",
    ),
    dict(type="PackOlmoEarthSegInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(crop_size, crop_size), keep_ratio=True),
    dict(type="LoadAnnotations"),
    dict(
        type="RGBToOlmoEarthS2",
        num_timesteps=num_timesteps,
        rgb_channel_order="BGR",
        input_value_range="0_255",
    ),
    dict(type="PackOlmoEarthSegInputs"),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(pipeline=train_pipeline),
)

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

data_preprocessor = dict(
    _delete_=True,
    type="OlmoEarthSegDataPreProcessor",
    mean=None,
    std=None,
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=ignore_index,
    size=(crop_size, crop_size),
    test_cfg=dict(size_divisor=patch_size),
)

model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        _delete_=True,
        type="UPerHead",
        in_channels=[768],
        in_index=[0],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1.0,
        ),
    ),
    auxiliary_head=dict(
        type="FCNHead",
        in_channels=768,
        in_index=0,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=0.4,
        ),
    ),
    test_cfg=dict(mode="whole"),
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="SGD",
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0005,
    ),
    clip_grad=None,
)

param_scheduler = [
    dict(
        type="PolyLR",
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=80000,
        by_epoch=False,
    ),
]

train_cfg = dict(type="IterBasedTrainLoop", max_iters=80000, val_interval=8000)

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=8000,
        save_best="mIoU",
    ),
    visualization=dict(type="OlmoEarthVisualizationHook"),
)
