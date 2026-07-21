custom_imports = dict(
    imports=["projects.tessera.tessera"],
    allow_failed_imports=False,
)

# Expected manifest layout:
# {
#   "samples": [
#     {
#       "tile_path": "retiled_d_pixel/tile_001",
#       "seg_map_path": "labels/tile_001_label.tif"
#     }
#   ]
# }
# Each tile_path should contain the standard TESSERA preprocessing files:
# bands.npy, masks.npy, doys.npy, sar_ascending.npy, sar_ascending_doy.npy,
# sar_descending.npy, sar_descending_doy.npy.
data_root = "/absolute/path/to/tessera_temporal_dataset"
tessera_checkpoint = "/absolute/path/to/tessera/checkpoints/best_model_fsdp_20250427_084307.pt"
work_dir = "./work_dirs/tessera_1xb2-50e_crop-type-online-linear"

ignore_index = 255
num_classes = 10
sample_size_s2 = 40
sample_size_s1 = 40
hidden_dim = 128
crop_size = (128, 128)

train_pipeline = [
    dict(
        type="LoadTesseraTemporalArrays",
        sample_size_s2=sample_size_s2,
        sample_size_s1=sample_size_s1,
        random_sample=True,
        standardize=True,
        ignore_index=ignore_index,
    ),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=1.0),
    dict(type="PackTesseraSegInputs"),
]

test_pipeline = [
    dict(
        type="LoadTesseraTemporalArrays",
        sample_size_s2=sample_size_s2,
        sample_size_s1=sample_size_s1,
        random_sample=False,
        standardize=True,
        ignore_index=ignore_index,
    ),
    dict(type="PackTesseraSegInputs"),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="TesseraSegDataset",
        data_root=data_root,
        ann_file="train.json",
        dataset_name="crop_type",
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="TesseraSegDataset",
        data_root=data_root,
        ann_file="val.json",
        dataset_name="crop_type",
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="TesseraSegDataset",
        data_root=data_root,
        ann_file="test.json",
        dataset_name="crop_type",
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

val_evaluator = dict(
    type="IoUMetric",
    ignore_index=ignore_index,
    iou_metrics=["mIoU"],
)
test_evaluator = val_evaluator

data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=None,
    std=None,
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=ignore_index,
    size=crop_size,
)

model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="TesseraBackbone",
        sample_size_s2=sample_size_s2,
        sample_size_s1=sample_size_s1,
        latent_dim=hidden_dim,
        out_channels=hidden_dim,
        fusion_method="concat",
        chunk_size=4096,
        frozen=True,
        init_cfg=dict(type="Pretrained", checkpoint=tessera_checkpoint),
    ),
    decode_head=dict(
        type="TesseraLinearHead",
        in_channels=hidden_dim,
        channels=hidden_dim,
        in_index=0,
        num_classes=num_classes,
        scale_factor=1,
        ignore_index=ignore_index,
        align_corners=False,
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
    logger=dict(type="LoggerHook", interval=20, log_metric_by_epoch=True),
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
