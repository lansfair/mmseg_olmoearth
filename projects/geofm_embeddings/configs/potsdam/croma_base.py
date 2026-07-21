_base_ = "./_base_normalized.py"

model = dict(
    backbone=dict(
        adapter=dict(
            _delete_=True,
            type="OfficialOlmoEarthWrapperAdapter",
            preset="croma",
            model_variant="base",
            modalities=["sentinel2_l2a"],
            out_channels=768,
            native_stride=8,
            wrapper_kwargs=dict(
                size="base",
                load_directory=(
                    "/mnt/ht2-nas2/EO_test/wyf/embedding_code/"
                    "地球基础模型权重/geofm/croma/base"
                ),
            ),
            freeze=True,
        )
    )
)
