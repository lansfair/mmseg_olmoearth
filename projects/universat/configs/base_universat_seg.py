"""Base MMSegmentation 1.x config for UniverSat + segmentation head.

This is a generic template. Replace the dataset paths, normalization statistics,
``num_classes`` and ``ignore_index`` with the values of your actual dataset.

Run from the MMSegmentation root (the folder that contains ``tools/``)::

    export PYTHONPATH=".:$PYTHONPATH"
    python tools/train.py \
        projects/universat/configs/base_universat_seg.py \
        --work-dir work_dirs/base_universat_seg
"""

_base_ = [
    "../../../configs/_base_/default_runtime.py",
]

custom_imports = dict(
    imports=["projects.universat.universat"],
    allow_failed_imports=False,
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
dataset_type = "UniverSatSegDataset"
data_root = "data/your_eo_seg_dataset/"

modalities = ["s2", "s1"]
crop_size = (360, 360)
num_classes = 10
ignore_index = 255

norm_cfg = dict(
    mean={
        "s2": [0.0] * 10,
        "s1": [0.0] * 3,
    },
    std={
        "s2": [1.0] * 10,
        "s1": [1.0] * 3,
    },
)

train_pipeline = [
    dict(type="LoadMultimodalFromFile", modalities=modalities),
    dict(type="LoadAnnotations"),
    dict(type="NormalizeMultimodal", **norm_cfg),
    dict(type="PackUniverSatInputs"),
]

test_pipeline = [
    dict(type="LoadMultimodalFromFile", modalities=modalities),
    dict(type="LoadAnnotations"),
    dict(type="NormalizeMultimodal", **norm_cfg),
    dict(type="PackUniverSatInputs"),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        modalities=modalities,
        data_root=data_root,
        split="splits/train.json",
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        modalities=modalities,
        data_root=data_root,
        split="splits/val.json",
        pipeline=test_pipeline,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type="IoUMetric",
    iou_metrics=["mIoU", "mFscore"],
    ignore_index=ignore_index,
)
test_evaluator = val_evaluator

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
backbone_embed_dim = 768

model = dict(
    type="EncoderDecoder",
    data_preprocessor=dict(type="UniverSatDataPreprocessor"),
    backbone=dict(
        type="UniverSatBackbone",
        modalities=modalities,
        embed_dim=backbone_embed_dim,
        num_heads=12,
        patch_size=40,
        output_grid=36,
        block_type=("Bi_ACA_in", "SAx12", "Bilinear_out", "CA_Sub"),
        n_registers=4,
        gating=True,
        compile_encoder=False,
        freeze_backbone=False,
        frozen_stages=-1,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="path/to/universat_base.safetensors",
        ),
    ),
    decode_head=dict(
        type="UniverSatSegHead",
        in_channels=backbone_embed_dim,
        in_index=0,
        channels=256,
        num_convs=2,
        output_size=crop_size,
        num_classes=num_classes,
        ignore_index=ignore_index,
        norm_cfg=dict(type="BN", requires_grad=True),
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

# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=50, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.05),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(
        type="OneCycleLR",
        eta_max=1e-4,
        pct_start=0.0,
        anneal_strategy="cos",
        begin=0,
        end=50,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

log_processor = dict(by_epoch=True)

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", by_epoch=True, interval=1),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)
