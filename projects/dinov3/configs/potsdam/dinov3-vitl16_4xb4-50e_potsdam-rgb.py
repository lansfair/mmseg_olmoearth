custom_imports = dict(
    imports=[
        "projects.dinov3.mmseg_dinov3",
        "projects.olmoearth.olmoearth",
    ],
    allow_failed_imports=False,
)

data_root = "/mnt/ht2-nas2/EO_test/mty/potsdam"
dinov3_root = "/mnt/ht2-nas2/EO_test/dataset/dinov3_pretrained"
dinov3_repo_dir = "/mnt/ht2-nas2/wj/large_tif_infer_test/mmseg_olmoearth/projects/dinov3/dinov3-main"
dinov3_weights_path = (
    f"{dinov3_root}/DINOv3 ViT SAT-493M/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
)
work_dir = "./work_dirs/dinov3-vitl16_4xb4-50e_potsdam-rgb-rvsa"

ignore_index = 5
num_classes = 5
crop_size = 256
patch_size = 16
hidden_dim = 1024

train_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
    dict(type="LoadAnnotations"),
    dict(
        type="RandomCrop",
        crop_size=(crop_size, crop_size),
        cat_max_ratio=0.75,
    ),
    dict(
        type="RandomRotate",
        prob=0.5,
        degree=90,
        pad_val=0,
        seg_pad_val=ignore_index,
    ),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="RandomFlip", prob=0.5, direction="vertical"),
    dict(type="PackSegInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="OlmoEarthPotsdamDataset",
        data_root=data_root,
        data_prefix=dict(
            img_path="img_dir/train",
            seg_map_path="ann_dir/train",
        ),
        label_mapping="official_to_rvsa_class5_ignore5",
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="OlmoEarthPotsdamDataset",
        data_root=data_root,
        data_prefix=dict(
            img_path="img_dir/val",
            seg_map_path="ann_dir/val",
        ),
        label_mapping="official_to_rvsa_class5_ignore5",
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

data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[109.65, 104.805, 75.48],
    std=[54.315, 39.78, 36.465],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=ignore_index,
    size_divisor=patch_size,
    test_cfg=dict(size_divisor=patch_size),
)

model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="DINOv3ViTBackbone",
        repo_dir=dinov3_repo_dir,
        model_name="dinov3_vitl16",
        weights_path=dinov3_weights_path,
        patch_size=patch_size,
        out_channels=hidden_dim,
        freeze=True,
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
    train_cfg=dict(),
    test_cfg=dict(
        mode="slide",
        crop_size=(crop_size, crop_size),
        stride=(crop_size // 2, crop_size // 2),
    ),
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.01, weight_decay=0.01),
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1e-6,
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

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=50, val_interval=5)
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
auto_scale_lr = dict(enable=False, base_batch_size=16)
