custom_imports = dict(
    imports=["projects.olmoearth.olmoearth"],
    allow_failed_imports=False,
)

data_root = "data/olmoearth_mmseg/pastis"
olmoearth_model_dir = "checkpoints/olmoearth"
model_config_path = f"{olmoearth_model_dir}/config.json"
weights_path = f"{olmoearth_model_dir}/weights.pth"
work_dir = "./work_dirs/olmoearth-base_4xb8-50e_pastis-s2-amp-ft"

ignore_index = 255
num_classes = 19
num_timesteps = 12
crop_size = (64, 64)
patch_size = 4
hidden_dim = 768

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
    dict(type="OlmoEarthRandomFlip", horizontal=True, vertical=True),
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
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="OlmoEarthSegDataset",
        data_root=data_root,
        ann_file="train.json",
        dataset_name="pastis",
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="OlmoEarthSegDataset",
        data_root=data_root,
        ann_file="val.json",
        dataset_name="pastis",
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

test_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="OlmoEarthSegDataset",
        data_root=data_root,
        ann_file="test.json",
        dataset_name="pastis",
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
        out_channels=hidden_dim,
        pooling_type="mean",
        fast_pass=True,
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

# Full fine-tuning: no FreezeBackboneUntilEpochHook here.
custom_hooks = []

optim_wrapper = dict(
    type="AmpOptimWrapper",
    loss_scale="dynamic",
    optimizer=dict(type="AdamW", lr=1e-3, weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1),
            decode_head=dict(lr_mult=1.0),
        )
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
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
        eta_min=1e-6,
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
log_processor = dict(by_epoch=True)
log_level = "INFO"
load_from = None
resume = False
tta_model = None
auto_scale_lr = dict(enable=False, base_batch_size=32)
