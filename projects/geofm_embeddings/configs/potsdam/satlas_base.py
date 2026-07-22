_base_ = "./_base_reflectance.py"

model = dict(
    backbone=dict(
        adapter=dict(
            _delete_=True,
            type="OfficialOlmoEarthWrapperAdapter",
            preset="satlas",
            model_variant="base",
            modalities=["sentinel2_l2a"],
            out_channels=1024,
            native_stride=8,
            wrapper_kwargs=dict(
                load_directory=(
                    "/mnt/ht2-nas2/EO_test/wyf/embedding_code/"
                    "地球基础模型权重/geofm/satlas/base"
                ),
                size="base",
                use_pretrained_normalizer=True,
            ),
            freeze=True,
        )
    )
)
