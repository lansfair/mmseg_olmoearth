custom_imports = dict(
    imports=["projects.olmoearth.olmoearth"],
    allow_failed_imports=False,
)

geobench_root = "/mnt/ht2-nas2/EO_test/dataset/geo-bench-1.0"
olmoearth_model_dir = "/mnt/ht2-nas2/EO_test/model/OlmoEarth-v1-Base"
model_config_path = f"{olmoearth_model_dir}/config.json"
weights_path = f"{olmoearth_model_dir}/weights.pth"
work_dir = "./work_dirs/olmoearth-base_1xb8-50e_m-cashew-plant-s2-linear"

ignore_index = 255
num_classes = 7
num_timesteps = 1
crop_size = (256, 256)
patch_size = 4
hidden_dim = 768

s2_band_names = [
    "02",
    "03",
    "04",
    "08",
    "05",
    "06",
    "07",
    "08A",
    "11",
    "12",
    "01",
    "09",
]

geobench_s2_imputes = [("11 - SWIR", "10 - SWIR - Cirrus")]

train_pipeline = [
    dict(
        type="LoadGeoBenchS2OfficialNorm",
        num_classes=num_classes,
        ignore_index=ignore_index,
        invalid_label_to_ignore=True,
        imputes=geobench_s2_imputes,
        default_timestamp=(15, 4, 2024),
    ),
    dict(
        type="RandomCrop",
        crop_size=crop_size,
        cat_max_ratio=1.0,
    ),
    dict(type="PackOlmoEarthSegInputs"),
]

test_pipeline = [
    dict(
        type="LoadGeoBenchS2OfficialNorm",
        num_classes=num_classes,
        ignore_index=ignore_index,
        invalid_label_to_ignore=True,
        imputes=geobench_s2_imputes,
        default_timestamp=(15, 4, 2024),
    ),
    dict(type="PackOlmoEarthSegInputs"),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="GeoBenchS2SegDataset",
        task_name="m-cashew-plant",
        benchmark_name="segmentation_v1.0",
        split="train",
        partition_name="default",
        band_names=s2_band_names,
        geobench_format="hdf5",
        geobench_root=geobench_root,
        dataset_name="cashew_plant",
        num_classes=num_classes,
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="GeoBenchS2SegDataset",
        task_name="m-cashew-plant",
        benchmark_name="segmentation_v1.0",
        split="valid",
        partition_name="default",
        band_names=s2_band_names,
        geobench_format="hdf5",
        geobench_root=geobench_root,
        dataset_name="cashew_plant",
        num_classes=num_classes,
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

test_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="GeoBenchS2SegDataset",
        task_name="m-cashew-plant",
        benchmark_name="segmentation_v1.0",
        split="test",
        partition_name="default",
        band_names=s2_band_names,
        geobench_format="hdf5",
        geobench_root=geobench_root,
        dataset_name="cashew_plant",
        num_classes=num_classes,
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

val_evaluator = dict(
    type="OlmoEarthIoUMetric",
    num_classes=num_classes,
    ignore_index=ignore_index,
    iou_metrics=["mIoU"],
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
    size=crop_size,
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
        out_channels=hidden_dim,
        pooling_type="mean",
    ),
    decode_head=dict(
        type="OlmoEarthPatchLinearHead",
        in_channels=hidden_dim,
        channels=hidden_dim,
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
    auxiliary_head=None,
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

custom_hooks = [dict(type="FreezeBackboneUntilEpochHook", unfreeze_epoch=None)]

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.1, weight_decay=0.0),
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1e-3,
        begin=0,
        end=5,
        by_epoch=True,
    ),
    dict(
        type="CosineAnnealingLR",
        eta_min=1e-5,
        begin=5,
        end=50,
        T_max=45,
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
auto_scale_lr = dict(enable=False, base_batch_size=8)
