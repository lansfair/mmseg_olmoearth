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
                    "/mnt/ht2-nas2/EO_test/miniconda3/envs/"
                    "geofm-olmoearth-cu121/lib/python3.11/site-packages/"
                    "olmoearth_pretrain/evals/models/clay/metadata.yaml"
                ),
                use_pretrained_normalizer=True,
            ),
            freeze=True,
        )
    )
)
