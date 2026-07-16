_base_ = [
    "../../../olmoearth/configs/pastis/"
    "olmoearth-base_4xb4-50e_pastis-s2.py"
]

custom_imports = dict(
    imports=[
        "projects.olmoearth.olmoearth",
        "projects.geofm_embeddings.geofm_embeddings",
    ],
    allow_failed_imports=False,
)

model = dict(
    type="GeoFMEncoderDecoder",
    backbone=dict(
        _delete_=True,
        type="GeoFMBackbone",
        output_mode="dense",
        frozen=True,
        adapter=dict(
            type="OlmoEarthAdapter",
            model_config_path=(
                "checkpoints/geofm/olmoearth/base/config.json"
            ),
            init_cfg=dict(
                type="Pretrained",
                checkpoint=(
                    "checkpoints/geofm/olmoearth/base/weights.pth"
                ),
            ),
            model_variant="base",
            modalities=["sentinel2_l2a"],
            num_timesteps=12,
            patch_size=4,
            pooling_type="mean",
            out_channels=768,
            freeze=True,
        ),
    ),
    decode_head=dict(
        type="GeoFMPatchLinearHead",
        in_channels=768,
        channels=768,
        in_index=0,
        num_classes=19,
        patch_size=4,
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

custom_hooks = [
    dict(
        type="GeoFMFreezeBackboneHook",
        unfreeze_epoch=None,
        strict=True,
    )
]
