_base_ = "./dinov3-vitl16_pastis-10band-43t-128x128.py"

work_dir = "./work_dirs/dinov3-vitl16_scale-pastis-10band-43t-128x128"

train_pipeline = [
    dict(
        type="DINOv3PASTISS2Normalize",
        num_timesteps=43,
        num_bands=10,
    ),
    dict(type="OlmoEarthRandomFlip", horizontal=True, vertical=True),
    dict(type="PackOlmoEarthSegInputs"),
]

test_pipeline = [
    dict(
        type="DINOv3PASTISS2Normalize",
        num_timesteps=43,
        num_bands=10,
    ),
    dict(type="PackOlmoEarthSegInputs"),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
