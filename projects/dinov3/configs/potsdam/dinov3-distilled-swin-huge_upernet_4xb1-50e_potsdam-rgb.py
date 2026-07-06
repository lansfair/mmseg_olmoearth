_base_ = "./dinov3-vitl16_4xb4-50e_potsdam-rgb.py"

work_dir = "./work_dirs/dinov3-distilled-swin-huge_upernet_4xb1-50e_potsdam-rgb-rvsa"

ignore_index = 5
num_classes = 5
crop_size = 256
swin_huge_channels = [352, 704, 1408, 2816]
norm_cfg = dict(type="SyncBN", requires_grad=True)

# Accept either an extracted pure backbone state_dict or the merged DINOv3
# distillation checkpoint. The wrapper prefers model_ema.backbone weights.
swin_huge_checkpoint = (
    "/mnt/si000523ygkv/00-model/dinov3-distill-outputs/"
    "swin_base_vitl16_ssl_feature_distill_GE+IN22k+ZJSlice1024_16nodes_nowarmup_lowlr/"
    "ckpt/30999/swintransformer-huge.pt"
)

train_dataloader = dict(
    batch_size=1,
    dataset=dict(label_mapping="official_to_rvsa_class5_ignore5"),
)

val_dataloader = dict(
    dataset=dict(label_mapping="official_to_rvsa_class5_ignore5"),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="OlmoEarthIoUMetric",
    num_classes=num_classes,
    ignore_index=ignore_index,
    use_valid_mask=False,
)
test_evaluator = val_evaluator

model = dict(
    data_preprocessor=dict(
        seg_pad_val=ignore_index,
        size_divisor=32,
        test_cfg=dict(size_divisor=32),
    ),
    backbone=dict(
        _delete_=True,
        type="DINOv3DistilledSwinHuge",
        checkpoint=swin_huge_checkpoint,
        img_size=crop_size,
        patch_size=4,
        window_size=8,
        out_indices=(0, 1, 2, 3),
        use_ema=True,
        frozen=False,
    ),
    decode_head=dict(
        _delete_=True,
        type="UPerHead",
        in_channels=swin_huge_channels,
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        ignore_index=ignore_index,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1.0,
        ),
    ),
    auxiliary_head=dict(
        type="FCNHead",
        in_channels=1408,
        in_index=2,
        channels=512,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        ignore_index=ignore_index,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=0.4,
        ),
    ),
    test_cfg=dict(
        mode="slide",
        crop_size=(crop_size, crop_size),
        stride=(crop_size // 2, crop_size // 2),
    ),
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.05),
    clip_grad=dict(max_norm=1.0, norm_type=2),
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
        eta_min=1e-6,
        begin=5,
        end=50,
        T_max=45,
        by_epoch=True,
    ),
]

auto_scale_lr = dict(enable=False, base_batch_size=4)
