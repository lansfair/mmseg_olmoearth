"""UniverSat Base + fine-tuning on PASTIS-R.

Run from the MMSegmentation root (the folder that contains ``tools/``)::

    export PYTHONPATH=".:$PWD/projects/universat:$PWD/projects/universat/pastis:$PYTHONPATH"
    python tools/train.py \
        projects/universat/pastis/configs/universat-base_pastis_ft.py \
        --work-dir work_dirs/universat-base_pastis_ft
"""

custom_imports = dict(
    imports=[
        "projects.universat.universat",
        "projects.universat.pastis.universat_pastis",
    ],
    allow_failed_imports=False,
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
data_root = "{{$MM_ARCHIVE_DATA_HOME:data}}/PASTIS-R"
pretrained = "{{$MM_ARCHIVE_CKPT_HOME:checkpoints}}/model.safetensors"


crop_size = (128, 128)
num_classes = 19
ignore_index = 19  # void class; background (0) is a valid class

modalities = ["s2", "s1"]

train_pipeline = [
    dict(type="PackUniverSatPASTISInputs", modalities=modalities),
]
test_pipeline = train_pipeline

train_dataloader = dict(
    # Dense 128 x 128 features are memory intensive. Use gradient
    # accumulation when a larger effective batch is required.
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    collate_fn=dict(type="universat_pastis_collate"),
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="UniverSatPASTISDataset",
        data_root=data_root,
        modalities=modalities,
        folds=[1, 2, 3],
        norm_path=".",
        temporal_dropout=200,
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    collate_fn=dict(type="universat_pastis_collate"),
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="UniverSatPASTISDataset",
        data_root=data_root,
        modalities=modalities,
        folds=[4],
        norm_path=".",
        temporal_dropout=200,  # official code also caps validation series
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    collate_fn=dict(type="universat_pastis_collate"),
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="UniverSatPASTISDataset",
        data_root=data_root,
        modalities=modalities,
        folds=[5],
        norm_path=".",
        temporal_dropout=0,
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

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
        output_grid=128,
        block_type=("Bi_ACA_in", "SAx12", "Bilinear_out", "CA_Sub"),
        n_registers=4,
        gating=True,
        compile_encoder=False,
        freeze_backbone=False,
        frozen_stages=-1,  # full fine-tune
        init_cfg=dict(
            type="Pretrained",
            checkpoint=pretrained,
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
max_epochs = 50

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    type="AmpOptimWrapper",
    loss_scale="dynamic",
    optimizer=dict(type="AdamW", lr=2e-4, weight_decay=0.05),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=500,
    ),
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=1.0,
        begin=0,
        end=max_epochs,
        by_epoch=True,
    ),
]

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
default_scope = "mmseg"
log_processor = dict(by_epoch=True)
log_level = "INFO"
load_from = None
resume = False

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=True,
        interval=1,
        save_best="mIoU",
        rule="greater",
        max_keep_ckpts=3,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)

randomness = dict(seed=0, deterministic=False)
