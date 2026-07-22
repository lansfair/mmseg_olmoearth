_base_ = "./_base_reflectance.py"

custom_imports = dict(
    imports=[
        "projects.geofm_embeddings.geofm_embeddings",
        "projects.universat.universat",
    ],
    allow_failed_imports=False,
)

model = dict(
    backbone=dict(
        adapter=dict(
            _delete_=True,
            type="UniverSatAdapter",
            model_variant="base",
            modalities=["sentinel2_l2a"],
            patch_size=40.0,
            output_grid=64,
            out_channels=768,
            s2_input_order="olmoearth",
            model_cfg=dict(
                type="UniverSatBackbone",
                modalities=["s2"],
                embed_dim=768,
                num_heads=12,
                patch_size=40.0,
                output_grid=64,
                freeze_backbone=True,
                init_cfg=dict(
                    checkpoint=(
                        "/mnt/ht2-nas2/EO_test/wyf/embedding_code/"
                        "地球基础模型权重/geofm/universat/model.safetensors"
                    )
                ),
            ),
            freeze=True,
        )
    )
)
