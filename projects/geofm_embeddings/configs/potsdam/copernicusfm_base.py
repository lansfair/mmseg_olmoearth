_base_ = "./_base_normalized.py"

model = dict(
    backbone=dict(
        adapter=dict(
            _delete_=True,
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
                    "/mnt/ht2-nas2/EO_test/wyf/embedding_code/地球基础模型权重/"
                    "geofm/copernicusfm/base/CopernicusFM_ViT_base_varlang_e100.pth"
                ),
            ),
            freeze=True,
        )
    )
)
