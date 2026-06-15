# Multi-GPU
model_wrapper_cfg = None
find_unused_parameters = True


custom_imports = dict(
    imports=["projects.olmoearth.olmoearth"],
    allow_failed_imports=False,
)


data_root = "/mnt/ht2-nas2/EO_test/wj1/PASTIS_evel/dataset/pastis_dataset_for_mmolmo"
olmoearth_model_dir = "../../../../PASTIS_evel/OEF/olmoearth_weights_base"
model_config_path = f"{olmoearth_model_dir}/config.json"
weights_path = f"{olmoearth_model_dir}/weights.pth"
work_dir = "./work_dirs/olmoearth-base_4xb4-50e_pastis-s2-plateau"


ignore_index = 255
num_classes = 19
num_timesteps = 12
crop_size = (64, 64)
patch_size = 4


train_pipeline = [
    dict(
        type="LoadOlmoEarthArrays",
        ignore_index=ignore_index,
        source_ignore_values=(-1,),
    ),
    dict(
        type="OlmoEarthNormalize",
        modality="sentinel2_l2a",
        num_timesteps=num_timesteps,
    ),
    dict(
        type="OlmoEarthRandomFlip",
        horizontal=True,
        vertical=True,
    ),
    dict(type="PackOlmoEarthSegInputs"),
]


test_pipeline = [
    dict(
        type="LoadOlmoEarthArrays",
        ignore_index=ignore_index,
        source_ignore_values=(-1,),
    ),
    dict(
        type="OlmoEarthNormalize",
        modality="sentinel2_l2a",
        num_timesteps=num_timesteps,
    ),
    dict(type="PackOlmoEarthSegInputs"),
]


train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        type="DefaultSampler",
        shuffle=True,
    ),
    dataset=dict(
        type="OlmoEarthSegDataset",
        data_root=data_root,
        ann_file="train.json",
        dataset_name="pastis",
        pipeline=train_pipeline,
    ),
)


val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        type="DefaultSampler",
        shuffle=False,
    ),
    dataset=dict(
        type="OlmoEarthSegDataset",
        data_root=data_root,
        ann_file="val.json",
        dataset_name="pastis",
        pipeline=test_pipeline,
    ),
)


test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        type="DefaultSampler",
        shuffle=False,
    ),
    dataset=dict(
        type="OlmoEarthSegDataset",
        data_root=data_root,
        ann_file="test.json",
        dataset_name="pastis",
        pipeline=test_pipeline,
    ),
)


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
    test_cfg=dict(
        size_divisor=patch_size,
    ),
)


model = dict(
    type="OlmoEarthEncoderDecoder",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="OlmoEarthBackbone",
        model_config_path=model_config_path,
        init_cfg=dict(
            type="Pretrained",
            checkpoint=weights_path,
        ),
        modality="sentinel2_l2a",
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
        mode="whole",
    ),
)


# Freeze the backbone during the first 10 epochs, then unfreeze it.
custom_hooks = [
    dict(
        type="FreezeBackboneUntilEpochHook",
        unfreeze_epoch=10,
    ),
]


# AdamW optimizer
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=5e-4,
        weight_decay=0.01,
    ),
)


# Plateau-based learning-rate schedule:
# - Monitor validation mIoU (larger is better).
# - Reduce the current learning rate to 0.2 times its previous value.
# - patience=1 follows MMEngine semantics and triggers the reduction after
#   the second consecutive validation epoch without improvement.
# - After a reduction, wait for a cooldown of 10 epochs before another
#   reduction is allowed.
param_scheduler = [
    dict(
        type="ReduceOnPlateauLR",
        monitor="mIoU",
        rule="greater",
        factor=0.2,
        patience=1,
        threshold=1e-4,
        threshold_rule="rel",
        cooldown=10,
        by_epoch=True,
        begin=0,
        end=50,
        verbose=True,
    ),
]


train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=50,
    val_interval=1,
)

val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")


visualizer = dict(
    type="SegLocalVisualizer",
    vis_backends=[
        dict(type="LocalVisBackend"),
    ],
    name="visualizer",
    alpha=1.0,
)


default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(
        type="LoggerHook",
        interval=50,
        log_metric_by_epoch=True,
    ),
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
    mp_cfg=dict(
        mp_start_method="fork",
        opencv_num_threads=0,
    ),
    dist_cfg=dict(
        backend="nccl",
    ),
)


default_scope = "mmseg"
log_level = "INFO"
load_from = None
resume = False
