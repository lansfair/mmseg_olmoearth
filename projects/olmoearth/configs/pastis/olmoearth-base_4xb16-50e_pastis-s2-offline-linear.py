custom_imports = dict(
    imports=["projects.olmoearth.olmoearth"],
    allow_failed_imports=False,
)

embedding_root = "work_dirs/olmoearth_embeddings/pastis_s2"
work_dir = "./work_dirs/olmoearth-base_4xb16-50e_pastis-s2-offline-linear"

ignore_index = 255
num_classes = 19
patch_size = 4
hidden_dim = 768
embedding_size = (16, 16)

train_pipeline = [
    dict(type="LoadOlmoEarthEmbedding", ignore_index=ignore_index),
    dict(type="PackOlmoEarthSegInputs"),
]

test_pipeline = train_pipeline

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="OlmoEarthSegDataset",
        data_root=embedding_root,
        ann_file="train.json",
        dataset_name="pastis",
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="OlmoEarthSegDataset",
        data_root=embedding_root,
        ann_file="val.json",
        dataset_name="pastis",
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

test_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="OlmoEarthSegDataset",
        data_root=embedding_root,
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
    size=embedding_size,
)

model = dict(
    type="OlmoEarthEncoderDecoder",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="OlmoEarthFeatureBackbone",
        out_channels=hidden_dim,
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
log_processor = dict(by_epoch=True)
log_level = "INFO"
load_from = None
resume = False
tta_model = None
auto_scale_lr = dict(enable=False, base_batch_size=16)
