_base_ = "./_base_reflectance.py"

model = dict(
    backbone=dict(
        adapter=dict(
            _delete_=True,
            type="OfficialOlmoEarthWrapperAdapter",
            preset="panopticon",
            model_variant="vitb14",
            modalities=["sentinel2_l2a"],
            out_channels=768,
            native_stride=14,
            external_source_path=(
                "/mnt/ht2-nas2/EO_test/wyf/embedding_code/geofm_a100/"
                "external/panopticon"
            ),
            local_checkpoint_path=(
                "/mnt/ht2-nas2/EO_test/wyf/embedding_code/"
                "地球基础模型权重/geofm/panopticon/vitb14/"
                "panopticon_vitb14_teacher.pth"
            ),
            wrapper_kwargs=dict(torchhub_id="panopticon_vitb14"),
            freeze=True,
        )
    )
)
