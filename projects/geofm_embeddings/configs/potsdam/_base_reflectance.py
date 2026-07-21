_base_ = "./_base_normalized.py"

reflectance_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
    dict(type="LoadAnnotations"),
    dict(type="ResizeImageOnly", size=64),
    dict(
        type="RGBToGeoFMS2",
        rgb_channel_order="BGR",
        input_value_range="0_255",
        representation="reflectance",
    ),
    dict(type="PackOlmoEarthSegInputs"),
]

train_dataloader = dict(dataset=dict(pipeline=reflectance_pipeline))
val_dataloader = dict(dataset=dict(pipeline=reflectance_pipeline))
test_dataloader = val_dataloader
model = dict(
    data_preprocessor=dict(input_representation="reflectance"),
)
