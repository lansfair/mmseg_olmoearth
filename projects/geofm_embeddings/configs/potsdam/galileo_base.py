_base_ = "./_base_reflectance.py"

model = dict(
    backbone=dict(
        adapter=dict(
            _delete_=True,
            type="OfficialOlmoEarthWrapperAdapter",
            preset="galileo",
            model_variant="base",
            modalities=["sentinel2_l2a"],
            out_channels=768,
            native_stride=4,
            wrapper_kwargs=dict(
                pretrained_path=(
                    "/mnt/ht2-nas2/EO_test/wyf/embedding_code/"
                    "地球基础模型权重/geofm/galileo/base"
                ),
                patch_size=4,
                use_pretrained_normalizer=True,
                autocast_dtype="bfloat16",
            ),
            freeze=True,
        )
    )
)
