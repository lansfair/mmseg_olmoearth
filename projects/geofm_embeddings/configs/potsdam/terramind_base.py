_base_ = "./_base_reflectance.py"

model = dict(
    backbone=dict(
        adapter=dict(
            _delete_=True,
            type="OfficialOlmoEarthWrapperAdapter",
            preset="terramind",
            model_variant="base",
            modalities=["sentinel2_l2a"],
            out_channels=768,
            native_stride=16,
            local_checkpoint_path=(
                "/mnt/ht2-nas2/EO_test/wyf/embedding_code/"
                "地球基础模型权重/geofm/terramind/base/"
                "TerraMind_v1_base.pt"
            ),
            wrapper_kwargs=dict(
                size="base",
                use_pretrained_normalizer=True,
            ),
            freeze=True,
        )
    )
)
