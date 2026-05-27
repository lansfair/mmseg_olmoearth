custom_imports = dict(
    imports=["projects.olmoearth.olmoearth"],
    allow_failed_imports=False,
)

data_root = "data/potsdam"
olmoearth_model_dir = "checkpoints/olmoearth"
model_config_path = f"{olmoearth_model_dir}/config.json"
weights_path = f"{olmoearth_model_dir}/weights.pth"
work_dir = "./work_dirs/olmoearth-base_4xb4-50e_potsdam-rgb"

ignore_index = 255
num_classes = 6
num_timesteps = 1
crop_size = 256
patch_size = 4

train_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
    dict(type="LoadAnnotations"),
    dict(
        type="RandomCrop",
        crop_size=(crop_size, crop_size),
        cat_max_ratio=0.75,
    ),
    dict(type="RandomFlip", prob=0.5),
    dict(
        type="RGBToOlmoEarthS2",
        num_timesteps=num_timesteps,
        rgb_channel_order="BGR",
        input_value_range="0_255",
    ),
    dict(type="PackOlmoEarthSegInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
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
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="OlmoEarthPotsdamDataset",
        data_root=data_root,
        data_prefix=dict(
            img_path="img_dir/train",
            seg_map_path="ann_dir/train",
        ),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="OlmoEarthPotsdamDataset",
        data_root=data_root,
        data_prefix=dict(
            img_path="img_dir/val",
            seg_map_path="ann_dir/val",
        ),
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="OlmoEarthIoUMetric",
    num_classes=num_classes,
    ignore_index=ignore_index,
    use_valid_mask=False,
)
test_evaluator = val_evaluator

data_preprocessor = dict(
    type="OlmoEarthSegDataPreProcessor",
    mean=None,
    std=None,
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=ignore_index,
    size_divisor=patch_size,
    test_cfg=dict(size_divisor=patch_size),
)

model = dict(
    type="OlmoEarthEncoderDecoder",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="OlmoEarthBackbone",
        model_config_path=model_config_path,
        init_cfg=dict(type="Pretrained", checkpoint=weights_path),
        modality="sentinel2_l2a",
        patch_size=patch_size,
        num_timesteps=num_timesteps,
        out_channels=768,
        pooling_type="mean",
    ),
    decode_head=dict(
        type="OlmoEarthPatchLinearHead",
        in_channels=768,
        channels=768,
        in_index=0,
        num_classes=num_classes,
        patch_size=patch_size,
        ignore_index=ignore_index,
        use_valid_mask=False,
        valid_mask_loss=False,
        align_corners=True,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1.0,
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(
        mode="slide",
        crop_size=(crop_size, crop_size),
        stride=(crop_size // 2, crop_size // 2),
    ),
)

custom_hooks = [dict(type="FreezeBackboneUntilEpochHook", unfreeze_epoch=None)]

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.1, weight_decay=0.01),
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1e-6,
        begin=0,
        end=5,
        by_epoch=True,
    ),
    dict(
        type="CosineAnnealingLR",
        eta_min=1e-5,
        begin=5,
        end=50,
        by_epoch=True,
    ),
]

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=50, val_interval=5)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=True,
        interval=5,
        save_best="mIoU",
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="OlmoEarthVisualizationHook"),
)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

default_scope = "mmseg"
log_level = "INFO"
load_from = None
resume = False
