custom_imports = dict(
    imports=["projects.olmoearth.olmoearth"],
    allow_failed_imports=False,
)

data_root = "data/olmoearth_mmseg/sen1floods11"
olmoearth_model_dir = "checkpoints/olmoearth"
model_config_path = f"{olmoearth_model_dir}/config.json"
weights_path = f"{olmoearth_model_dir}/weights.pth"
work_dir = "./work_dirs/olmoearth-base_4xb32-50e_sen1floods11-s1"

ignore_index = 255
num_classes = 2
num_timesteps = 1
patch_size = 4

train_pipeline = [
    dict(
        type="LoadOlmoEarthArrays",
        ignore_index=ignore_index,
    ),
    dict(
        type="OlmoEarthNormalize",
        modality="sentinel1",
        num_timesteps=num_timesteps,
    ),
    dict(type="OlmoEarthRandomFlip", horizontal=True, vertical=True),
    dict(type="PackOlmoEarthSegInputs"),
]

test_pipeline = [
    dict(
        type="LoadOlmoEarthArrays",
        ignore_index=ignore_index,
    ),
    dict(
        type="OlmoEarthNormalize",
        modality="sentinel1",
        num_timesteps=num_timesteps,
    ),
    dict(type="PackOlmoEarthSegInputs"),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="OlmoEarthSegDataset",
        data_root=data_root,
        ann_file="train.json",
        dataset_name="sen1floods11",
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="OlmoEarthSegDataset",
        data_root=data_root,
        ann_file="val.json",
        dataset_name="sen1floods11",
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
        model_config_path=model_config_path,
        init_cfg=dict(type="Pretrained", checkpoint=weights_path),
        modality="sentinel1",
        patch_size=patch_size,
        num_timesteps=num_timesteps,
        out_channels=768,
        pooling_type="mean",
        fast_pass=True,
    ),
    decode_head=dict(
        type="OlmoEarthPatchLinearHead",
        in_channels=768,
        channels=768,
        in_index=0,
        num_classes=num_classes,
        patch_size=patch_size,
        ignore_index=ignore_index,
        align_corners=True,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1.0,
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

custom_hooks = [dict(type="FreezeBackboneUntilEpochHook", unfreeze_epoch=10)]
optim_wrapper = dict(
    type="AmpOptimWrapper",
    loss_scale="dynamic",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.01),
)
param_scheduler = [
    dict(
        type="MultiStepLR",
        begin=0,
        end=50,
        milestones=[10],
        gamma=0.1,
        by_epoch=True,
    ),
    dict(
        type="ReduceOnPlateauParamScheduler",
        param_name="lr",
        monitor="mIoU",
        rule="greater",
        factor=0.2,
        patience=2,
        min_value=0.0,
        cooldown=10,
        by_epoch=True,
    ),
]
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=50, val_interval=10)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=True,
        interval=10,
        save_best="mIoU",
        rule="greater",
        max_keep_ckpts=3,
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
auto_scale_lr = dict(enable=False, base_batch_size=128)
