_base_ = "./olmoearth-base_upernet_4xb4-80k_potsdam-rgb-p4-512x512.py"

work_dir = (
    "./work_dirs/"
    "olmoearth-base_upernet_4xb4-80k_potsdam-rgb-native-p4-512x512"
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(
        type="RandomResize",
        scale=(crop_size, crop_size),
        ratio_range=(0.5, 2.0),
        keep_ratio=True,
    ),
    dict(
        type="RandomCrop",
        crop_size=(crop_size, crop_size),
        cat_max_ratio=0.75,
    ),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(
        type="RGBToOlmoEarthRGB",
        num_timesteps=num_timesteps,
        rgb_channel_order="BGR",
        input_value_range="0_255",
    ),
    dict(type="PackOlmoEarthSegInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(crop_size, crop_size), keep_ratio=True),
    dict(type="LoadAnnotations"),
    dict(
        type="RGBToOlmoEarthRGB",
        num_timesteps=num_timesteps,
        rgb_channel_order="BGR",
        input_value_range="0_255",
    ),
    dict(type="PackOlmoEarthSegInputs"),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

model = dict(backbone=dict(modality="rgb"))
