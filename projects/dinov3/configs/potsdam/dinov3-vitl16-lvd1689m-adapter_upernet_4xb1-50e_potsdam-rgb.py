_base_ = "./dinov3-vitl16-sat493m_4xb4-50e_potsdam-rgb.py"

work_dir = "./work_dirs/dinov3-vitl16-lvd1689m-adapter_upernet_4xb1-50e_potsdam-rgb-rvsa"

ignore_index = 5
num_classes = 5
crop_size = 256
hidden_dim = 1024
norm_cfg = dict(type="SyncBN", requires_grad=True)

dinov3_root = "/mnt/ht2-nas2/EO_test/dataset/dinov3_pretrained"
dinov3_weights_path = (
    f"{dinov3_root}/DINOv3 ViT LVD-1689M/"
    "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
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
        type="DINOv3AdapterBackbone",
        arch="vit_large",
        patch_size=16,
        weights_path=dinov3_weights_path,
        weight_variant="lvd1689m",
        freeze_vit=True,
        finetune_vit=False,
        replace_ms_deform_attn=True,
        with_cp=False,
    ),
    decode_head=dict(
        _delete_=True,
        type="UPerHead",
        in_channels=[hidden_dim, hidden_dim, hidden_dim, hidden_dim],
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
        in_channels=hidden_dim,
        in_index=2,
        channels=256,
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
