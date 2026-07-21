custom_imports = dict(
    imports=[
        "projects.geofm_embeddings.geofm_embeddings",
    ],
    allow_failed_imports=False,
)

data_root = "/mnt/ht2-nas2/EO_test/openmmlab-archive/dat/potsdam"
pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
    dict(type="LoadAnnotations"),
    dict(type="ResizeImageOnly", size=64),
    dict(
        type="RGBToGeoFMS2",
        rgb_channel_order="BGR",
        input_value_range="0_255",
        representation="normalized",
    ),
    dict(type="PackOlmoEarthSegInputs"),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="OlmoEarthPotsdamDataset",
        data_root=data_root,
        data_prefix=dict(
            img_path="img_dir/train", seg_map_path="ann_dir/train"
        ),
        pipeline=pipeline,
    ),
)
val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="OlmoEarthPotsdamDataset",
        data_root=data_root,
        data_prefix=dict(img_path="img_dir/val", seg_map_path="ann_dir/val"),
        pipeline=pipeline,
    ),
)
test_dataloader = val_dataloader

model = dict(
    type="GeoFMEmbeddingModel",
    data_preprocessor=dict(
        type="PotsdamGeoFMDataPreprocessor",
        input_representation="normalized",
    ),
    backbone=dict(
        type="GeoFMBackbone",
        output_mode="dense",
        frozen=True,
        adapter=dict(
            type="OlmoEarthAdapter",
            model_config_path=(
                "/mnt/ht2-nas2/EO_test/wyf/embedding_code/"
                "地球基础模型权重/geofm/olmoearth/base/config.json"
            ),
            init_cfg=dict(
                type="Pretrained",
                checkpoint=(
                    "/mnt/ht2-nas2/EO_test/wyf/embedding_code/"
                    "地球基础模型权重/geofm/olmoearth/base/weights.pth"
                ),
            ),
            model_variant="base",
            modalities=["sentinel2_l2a"],
            num_timesteps=1,
            patch_size=4,
            pooling_type="mean",
            out_channels=768,
            freeze=True,
        ),
    ),
)
default_scope = "mmseg"
