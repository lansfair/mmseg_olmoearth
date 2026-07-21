_base_ = "./_base_reflectance.py"

model = dict(
    data_preprocessor=dict(include_sentinel1=True),
    backbone=dict(
        adapter=dict(
            _delete_=True,
            type="TESSERAAdapter",
            num_timesteps=1,
            model_variant="v1",
            latent_dim=128,
            out_channels=128,
            use_pretrained_normalizer=True,
            chunk_size=8192,
            init_cfg=dict(
                type="Pretrained",
                checkpoint=(
                    "/mnt/ht2-nas2/EO_test/wyf/embedding_code/地球基础模型权重/"
                    "geofm/tessera/v1/best_model_fsdp_20250427_084307.pt"
                ),
            ),
            freeze=True,
        )
    ),
)
