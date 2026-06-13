_base_ = "./olmoearth-base_upernet_4xb4-80k_potsdam-rgb-p4-512x512.py"

work_dir = (
    "./work_dirs/"
    "olmoearth-base_upernet_4xb4-80k_potsdam-rgb-s2proxy-p4-512x512"
)

olmoearth_model_dir = "/mnt/ht2-nas2/EO_test/model/OlmoEarth-v1-Base"
model_config_path = f"{olmoearth_model_dir}/config.json"
weights_path = f"{olmoearth_model_dir}/weights.pth"

model = dict(
    backbone=dict(
        model_config_path=model_config_path,
        init_cfg=dict(type="Pretrained", checkpoint=weights_path),
        modality="sentinel2_l2a",
    )
)
