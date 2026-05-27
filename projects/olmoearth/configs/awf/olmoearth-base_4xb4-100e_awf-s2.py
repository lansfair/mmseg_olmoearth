custom_imports = dict(
    imports=["projects.olmoearth.olmoearth"],
    allow_failed_imports=False,
)

data_root = "data/olmoearth_mmseg/awf"
checkpoint_path = "checkpoints/olmoearth"
work_dir = "./work_dirs/olmoearth-base_4xb4-100e_awf-s2"

ignore_index = 255
num_classes = 10
num_timesteps = 12
patch_size = 4

train_pipeline = [
    dict(
        type="LoadOlmoEarthArrays",
        image_layout="CTHW",
        ignore_index=ignore_index,
    ),
    dict(type="OlmoEarthPad", size=31, mode="center"),
    dict(type="OlmoEarthCrop", crop_size=16, mode="random"),
    dict(type="OlmoEarthRandomFlip", horizontal=True, vertical=True),
    dict(
        type="OlmoEarthNormalize",
        modality="sentinel2_l2a",
        num_timesteps=num_timesteps,
    ),
    dict(type="PackOlmoEarthSegInputs"),
]

test_pipeline = [
    dict(
        type="LoadOlmoEarthArrays",
        image_layout="CTHW",
        ignore_index=ignore_index,
    ),
    dict(type="OlmoEarthPad", size=31, mode="center"),
    dict(type="OlmoEarthCrop", crop_size=16, mode="center"),
    dict(
        type="OlmoEarthNormalize",
        modality="sentinel2_l2a",
        num_timesteps=num_timesteps,
    ),
    dict(type="PackOlmoEarthSegInputs"),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="OlmoEarthSegDataset",
        data_root=data_root,
        ann_file="train.json",
        dataset_name="awf",
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="OlmoEarthSegDataset",
        data_root=data_root,
        ann_file="val.json",
        dataset_name="awf",
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="OlmoEarthAccuracyMetric",
    ignore_index=ignore_index,
    use_valid_mask=True,
)
test_evaluator = val_evaluator

model = dict(
    type="OlmoEarthEncoderDecoder",
    data_preprocessor=dict(
        type="OlmoEarthSegDataPreProcessor",
        mean=None,
        std=None,
        bgr_to_rgb=False,
        pad_val=0,
        seg_pad_val=ignore_index,
        size_divisor=patch_size,
        test_cfg=dict(size_divisor=patch_size),
    ),
    backbone=dict(
        type="OlmoEarthBackbone",
        checkpoint_path=checkpoint_path,
        modality="sentinel2_l2a",
        patch_size=patch_size,
        num_timesteps=num_timesteps,
        out_channels=768,
        pooling_type="mean",
    ),
    decode_head=dict(
        type="OlmoEarthLinearHead",
        in_channels=768,
        channels=768,
        in_index=0,
        num_classes=num_classes,
        scale_factor=patch_size,
        ignore_index=ignore_index,
        use_valid_mask=True,
        valid_mask_loss=True,
        align_corners=False,
        loss_decode=dict(
            type="ValidMaskCrossEntropyLoss",
            ignore_index=ignore_index,
            loss_weight=1.0,
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

custom_hooks = [dict(type="FreezeBackboneUntilEpochHook", unfreeze_epoch=20)]
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.0001),
    paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.1)}),
)
param_scheduler = [
    dict(
        type="ReduceOnPlateauLR",
        monitor="loss",
        rule="less",
        factor=0.2,
        patience=2,
        cooldown=10,
        min_value=0,
        by_epoch=True,
    )
]
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=100, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=True,
        interval=1,
        save_best="accuracy",
        rule="greater",
        save_last=True,
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
