custom_imports = dict(
    imports=[
        "projects.dinov3.dinov3",
        "projects.olmoearth.olmoearth",
    ],
    allow_failed_imports=False,
)

data_root = "/mnt/ht2-nas2/EO_test/wj1/PASTIS_evel/dataset/PASTIS-R"
pastis_norm_file = f"{data_root}/NORM_S2_patch.json"
dinov3_root = "/mnt/ht2-nas2/EO_test/dataset/dinov3_pretrained"
dinov3_repo_dir = "/mnt/ht2-nas2/EO_test/wyf/mmseg_olmoearth/projects/dinov3/dinov3-main"
dinov3_weights_path = (
    f"{dinov3_root}/DINOv3 ViT SAT-493M/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
)
work_dir = "./work_dirs/dinov3-vitl16_pastis-10band-12t-128x128"

ignore_index = 255
num_classes = 19
num_bands = 10
num_timesteps = 12
crop_size = (128, 128)
patch_size = 16
backbone_channels = 1024
decoder_channels = 256

train_pipeline = [
    dict(
        type="DINOv3PASTISS2Normalize",
        num_timesteps=num_timesteps,
        num_bands=num_bands,
        norm_file=pastis_norm_file,
        folds=(1, 2, 3),
    ),
    dict(type="OlmoEarthRandomFlip", horizontal=True, vertical=True),
    dict(type="PackOlmoEarthSegInputs"),
]

test_pipeline = [
    dict(
        type="DINOv3PASTISS2Normalize",
        num_timesteps=num_timesteps,
        num_bands=num_bands,
        norm_file=pastis_norm_file,
        folds=(1, 2, 3),
    ),
    dict(type="PackOlmoEarthSegInputs"),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="DINOv3RawPASTISDataset",
        data_root=data_root,
        folds=(1, 2, 3),
        num_timesteps=num_timesteps,
        num_bands=num_bands,
        ignore_index=ignore_index,
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="DINOv3RawPASTISDataset",
        data_root=data_root,
        folds=(4,),
        num_timesteps=num_timesteps,
        num_bands=num_bands,
        ignore_index=ignore_index,
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="DINOv3RawPASTISDataset",
        data_root=data_root,
        folds=(5,),
        num_timesteps=num_timesteps,
        num_bands=num_bands,
        ignore_index=ignore_index,
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
    test_cfg=dict(size_divisor=patch_size),
)

model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="DINOv3PASTISTemporalBackbone",
        repo_dir=dinov3_repo_dir,
        model_name="dinov3_vitl16",
        weights_path=dinov3_weights_path,
        num_bands=num_bands,
        num_timesteps=num_timesteps,
        patch_size=patch_size,
        in_size=224,
        out_indices=(8, 16, 23),
        backbone_channels=backbone_channels,
        out_channels=decoder_channels,
        tae_groups=16,
        freeze_dinov3=True,
    ),
    decode_head=dict(
        type="DINOv3PASTISUpHead",
        in_channels=decoder_channels,
        channels=decoder_channels,
        in_index=0,
        input_size=14,
        output_size=128,
        num_classes=num_classes,
        ignore_index=ignore_index,
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1.0,
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-3, weight_decay=1e-2),
    clip_grad=dict(max_norm=5.0, norm_type=2),
)

param_scheduler = [
    dict(
        type="CosineAnnealingLR",
        eta_min=1e-6,
        begin=0,
        end=100,
        T_max=100,
        by_epoch=True,
    ),
]

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=100, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

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
