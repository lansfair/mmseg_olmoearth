_base_ = "./dinov3-vitl16_pastis-10band-12t-128x128.py"

work_dir = "./work_dirs/dinov3-vitl16_linear-pastis-10band-12t-128x128"

model = dict(
    decode_head=dict(
        _delete_=True,
        type="DINOv3PASTISLinearHead",
        in_channels=256,
        channels=256,
        in_index=0,
        num_classes=19,
        patch_size=16,
        ignore_index=255,
        use_valid_mask=False,
        valid_mask_loss=False,
        align_corners=True,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1.0,
        ),
    ),
)
