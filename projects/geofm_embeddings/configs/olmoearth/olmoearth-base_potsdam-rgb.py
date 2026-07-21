_base_ = [
    "../../../olmoearth/configs/potsdam/"
    "olmoearth-base_4xb4-50e_potsdam-rgb.py"
]

custom_imports = dict(
    imports=[
        "projects.olmoearth.olmoearth",
        "projects.geofm_embeddings.geofm_embeddings",
    ],
    allow_failed_imports=False,
)

# Embedding export must be deterministic.  Reuse the evaluation transforms for
# train instead of the training config's RandomCrop/RandomFlip pipeline.
export_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
    dict(type="LoadAnnotations"),
    dict(
        type="RGBToOlmoEarthS2",
        num_timesteps=1,
        rgb_channel_order="BGR",
        input_value_range="0_255",
    ),
    dict(type="PackOlmoEarthSegInputs"),
]

train_dataloader = dict(
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(pipeline=export_pipeline),
)

model = dict(
    backbone=dict(
        _delete_=True,
        type="GeoFMBackbone",
        output_mode="dense",
        frozen=True,
        adapter=dict(
            type="OlmoEarthAdapter",
            model_config_path="checkpoints/geofm/olmoearth/base/config.json",
            init_cfg=dict(
                type="Pretrained",
                checkpoint="checkpoints/geofm/olmoearth/base/weights.pth",
            ),
            model_variant="base",
            modalities=["sentinel2_l2a"],
            num_timesteps=1,
            patch_size=4,
            pooling_type="mean",
            out_channels=768,
            freeze=True,
        ),
    )
)
