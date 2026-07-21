_base_ = "./_base_normalized.py"

model = dict(
    backbone=dict(
        adapter=dict(
            _delete_=True,
            type="DINOv3Adapter",
            repo_dir=(
                "/mnt/ht2-nas2/EO_test/wyf/embedding_code/geofm_a100/src/"
                "mmseg_olmoearth/projects/dinov3/dinov3-main"
            ),
            model_name="dinov3_vitl16",
            model_variant="vitl16-sat493m",
            weights_path=(
                "/mnt/ht2-nas2/EO_test/wyf/embedding_code/地球基础模型权重/"
                "geofm/dinov3/vitl16-sat/"
                "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
            ),
            patch_size=16,
            out_channels=1024,
            base_resize=256,
            apply_normalization=False,
            freeze=True,
        )
    )
)
