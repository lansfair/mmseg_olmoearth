_base_ = ["./olmoearth-base_pastis-s2.py"]

# Fair comparison track: every model uses one point-wise affine classifier.
# Bilinear interpolation has no trainable parameters.
model = dict(
    decode_head=dict(
        _delete_=True,
        type="GeoFMLinearHead",
        in_channels=768,
        channels=768,
        in_index=0,
        num_classes=19,
        scale_factor=4,
        ignore_index=255,
        use_valid_mask=False,
        valid_mask_loss=False,
        align_corners=True,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1.0,
        ),
    )
)
