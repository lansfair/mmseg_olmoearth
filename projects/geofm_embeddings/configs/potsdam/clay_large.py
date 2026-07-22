_base_ = "./_base_reflectance.py"

model = dict(
    backbone=dict(
        adapter=dict(
            _delete_=True,
            type="OfficialOlmoEarthWrapperAdapter",
            preset="clay",
            model_variant="large",
            modalities=["sentinel2_l2a"],
            out_channels=1024,
            native_stride=16,
            wrapper_kwargs=dict(
                size="large",
                load_path=(
                    "/mnt/ht2-nas2/EO_test/wyf/embedding_code/地球基础模型权重/"
                    "geofm/clay/large-v1.5/clay-v1.5.ckpt"
                ),
                metadata_path=(
                    "/mnt/ht2-nas2/EO_test/wyf/embedding_code/geofm_a100/"
                    "src/mmseg_olmoearth/projects/geofm_embeddings/"
                    "geofm_embeddings/adapters/clay_metadata.yaml"
                ),
                use_pretrained_normalizer=True,
            ),
            freeze=True,
        )
    )
)
