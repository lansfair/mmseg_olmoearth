custom_imports = dict(
    imports=["projects.olmoearth.olmoearth"],
    allow_failed_imports=False,
)

embedding_root = "work_dirs/olmoearth_embeddings/awf_s2"
work_dir = "./work_dirs/olmoearth-base_4xb16-100e_awf-s2-offline-linear"

ignore_index = 255
num_classes = 10
patch_size = 4
hidden_dim = 768
embedding_size = (4, 4)

train_pipeline = [
    dict(type="LoadOlmoEarthEmbedding", ignore_index=ignore_index),
    dict(type="PackOlmoEarthSegInputs"),
]

test_pipeline = train_pipeline

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="OlmoEarthSegDataset",
        data_root=embedding_root,
        ann_file="train.json",
        dataset_name="awf",
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="OlmoEarthSegDataset",
        data_root=embedding_root,
        ann_file="val.json",
        dataset_name="awf",
        pipeline=test_pipeline,
        test_mode=True,
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
        size=embedding_size,
    ),
    backbone=dict(
        type="OlmoEarthFeatureBackbone",
        out_channels=hidden_dim,
    ),
    decode_head=dict(
        type="OlmoEarthLinearHead",
        in_channels=hidden_dim,
        channels=hidden_dim,
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

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.0001),
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

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=100, val_interval=5)
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
        save_best="accuracy",
        rule="greater",
        max_keep_ckpts=3,
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
log_processor = dict(by_epoch=True)
log_level = "INFO"
load_from = None
resume = False
tta_model = None
auto_scale_lr = dict(enable=False, base_batch_size=16)
