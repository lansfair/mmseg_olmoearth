from __future__ import annotations

from copy import deepcopy


DATA_ROOT = "/mnt/ht2-nas2/EO_test/openmmlab-archive/dat/potsdam"
WEIGHT_ROOT = "/mnt/ht2-nas2/EO_test/wyf/embedding_code/地球基础模型权重/geofm"
REPO_ROOT = "/mnt/ht2-nas2/EO_test/wyf/embedding_code/geofm_a100/src/mmseg_olmoearth"
OLMO_PACKAGE = (
    "/mnt/ht2-nas2/EO_test/miniconda3/envs/geofm-olmoearth-cu121/"
    "lib/python3.11/site-packages/olmoearth_pretrain"
)


def _official(
    preset: str,
    out_channels: int,
    wrapper_kwargs: dict,
    *,
    native_stride: int | None = None,
    representation: str = "reflectance",
    modalities: tuple[str, ...] = ("sentinel2_l2a",),
) -> dict:
    return dict(
        representation=representation,
        adapter=dict(
            type="OfficialOlmoEarthWrapperAdapter",
            preset=preset,
            model_variant="base",
            modalities=list(modalities),
            out_channels=out_channels,
            native_stride=native_stride,
            wrapper_kwargs=wrapper_kwargs,
            freeze=True,
        ),
    )


MODEL_SPECS = {
    "olmoearth_base": dict(
        representation="normalized",
        adapter=dict(
            type="OlmoEarthAdapter",
            model_config_path=f"{WEIGHT_ROOT}/olmoearth/base/config.json",
            init_cfg=dict(
                type="Pretrained",
                checkpoint=f"{WEIGHT_ROOT}/olmoearth/base/weights.pth",
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
    "copernicusfm_base": dict(
        representation="normalized",
        adapter=dict(
            type="CopernicusFMAdapter",
            modalities=["sentinel2_l2a"],
            model_variant="base",
            temporal_pooling="mean",
            image_size=224,
            patch_size=16,
            out_channels=768,
            init_cfg=dict(
                type="Pretrained",
                checkpoint=(
                    f"{WEIGHT_ROOT}/copernicusfm/base/"
                    "CopernicusFM_ViT_base_varlang_e100.pth"
                ),
            ),
            freeze=True,
        ),
    ),
    "dinov3_vitl16": dict(
        representation="normalized",
        adapter=dict(
            type="DINOv3Adapter",
            repo_dir=f"{REPO_ROOT}/projects/dinov3/dinov3-main",
            model_name="dinov3_vitl16",
            model_variant="vitl16-sat493m",
            weights_path=(
                f"{WEIGHT_ROOT}/dinov3/vitl16-sat/"
                "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
            ),
            patch_size=16,
            out_channels=1024,
            base_resize=256,
            apply_normalization=False,
            freeze=True,
        ),
    ),
    "clay_large": _official(
        "clay",
        1024,
        dict(
            size="large",
            load_path=f"{WEIGHT_ROOT}/clay/large-v1.5/clay-v1.5.ckpt",
            metadata_path=f"{OLMO_PACKAGE}/evals/models/clay/metadata.yaml",
            use_pretrained_normalizer=True,
        ),
        native_stride=16,
    ),
    "croma_base": _official(
        "croma",
        768,
        dict(size="base", load_directory=f"{WEIGHT_ROOT}/croma"),
        native_stride=8,
        representation="normalized",
    ),
    "galileo_base": _official(
        "galileo",
        768,
        dict(
            pretrained_path=f"{WEIGHT_ROOT}/galileo/base",
            patch_size=4,
            use_pretrained_normalizer=True,
            autocast_dtype="bfloat16",
        ),
        native_stride=4,
    ),
    "presto": _official(
        "presto",
        128,
        dict(
            load_directory=f"{WEIGHT_ROOT}/presto/default",
            use_pretrained_normalizer=True,
        ),
        native_stride=1,
    ),
    "prithviv2_300m": _official(
        "prithviv2",
        1024,
        dict(
            load_directory=f"{WEIGHT_ROOT}/prithviv2",
            size="Prithvi-EO-2.0-300M",
            use_pretrained_normalizer=True,
        ),
        native_stride=16,
    ),
    "tessera": dict(
        representation="reflectance",
        include_sentinel1=True,
        adapter=dict(
            type="TESSERAAdapter",
            num_timesteps=1,
            model_variant="v1",
            latent_dim=128,
            out_channels=128,
            use_pretrained_normalizer=True,
            chunk_size=8192,
            init_cfg=dict(
                type="Pretrained",
                checkpoint=f"{WEIGHT_ROOT}/tessera/v1/best_model_fsdp_20250427_084307.pt",
            ),
            freeze=True,
        ),
    ),
}


def build_potsdam_config(model_name: str) -> dict:
    if model_name not in MODEL_SPECS:
        raise KeyError(
            f"Unknown Potsdam model {model_name!r}; choose from {sorted(MODEL_SPECS)}"
        )
    spec = deepcopy(MODEL_SPECS[model_name])
    representation = spec.pop("representation")
    include_sentinel1 = spec.pop("include_sentinel1", False)
    pipeline = [
        dict(type="LoadImageFromFile", to_float32=True),
        dict(type="LoadAnnotations"),
        dict(type="ResizeImageOnly", size=64),
        dict(
            type="RGBToGeoFMS2",
            rgb_channel_order="BGR",
            input_value_range="0_255",
            representation=representation,
        ),
        dict(type="PackOlmoEarthSegInputs"),
    ]

    def dataloader(split: str, batch_size: int) -> dict:
        return dict(
            batch_size=batch_size,
            num_workers=4,
            persistent_workers=True,
            sampler=dict(type="DefaultSampler", shuffle=False),
            dataset=dict(
                type="OlmoEarthPotsdamDataset",
                data_root=DATA_ROOT,
                data_prefix=dict(
                    img_path=f"img_dir/{split}",
                    seg_map_path=f"ann_dir/{split}",
                ),
                pipeline=deepcopy(pipeline),
            ),
        )

    return dict(
        custom_imports=dict(
            imports=[
                "projects.olmoearth.olmoearth",
                "projects.geofm_embeddings.geofm_embeddings",
            ],
            allow_failed_imports=False,
        ),
        model_name=model_name,
        model=dict(
            type="GeoFMEmbeddingModel",
            data_preprocessor=dict(
                type="PotsdamGeoFMDataPreprocessor",
                include_sentinel1=include_sentinel1,
                input_representation=representation,
            ),
            backbone=dict(
                type="GeoFMBackbone",
                output_mode="dense",
                frozen=True,
                adapter=spec["adapter"],
            ),
        ),
        train_dataloader=dataloader("train", batch_size=8),
        val_dataloader=dataloader("val", batch_size=8),
        test_dataloader=dataloader("val", batch_size=8),
        default_scope="mmseg",
    )
