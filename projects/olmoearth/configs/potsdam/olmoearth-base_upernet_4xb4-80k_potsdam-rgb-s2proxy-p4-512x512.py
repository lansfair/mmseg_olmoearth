_base_ = "./olmoearth-base_upernet_4xb4-80k_potsdam-rgb-p4-512x512.py"

work_dir = (
    "./work_dirs/"
    "olmoearth-base_upernet_4xb4-80k_potsdam-rgb-s2proxy-p4-512x512"
)

model = dict(backbone=dict(modality="sentinel2_l2a"))
