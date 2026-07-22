_base_ = "./_base_reflectance.py"

model = dict(
    backbone=dict(
        adapter=dict(
            _delete_=True,
            type="OfficialOlmoEarthWrapperAdapter",
            preset="anysat",
            model_variant="base",
            modalities=["sentinel2_l2a"],
            out_channels=768,
            native_stride=4,
            external_source_path=(
                "/mnt/ht2-nas2/EO_test/wyf/embedding_code/geofm_a100/"
                "external/AnySat"
            ),
            local_checkpoint_path=(
                "/mnt/ht2-nas2/EO_test/wyf/embedding_code/"
                "地球基础模型权重/geofm/anysat/base/AnySat.pth"
            ),
            wrapper_kwargs=dict(patch_size=4),
            freeze=True,
        )
    )
)
