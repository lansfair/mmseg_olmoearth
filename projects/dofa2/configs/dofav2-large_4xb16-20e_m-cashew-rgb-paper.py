_base_ = ['./dofav2-large_4xb16-20e_m-cashew-s2-paper.py']

rgb_band_names = (
    '04 - Red',
    '03 - Green',
    '02 - Blue',
)

model = dict(
    backbone=dict(
        model_bands=['RED', 'GREEN', 'BLUE'],
        convert_patch_14_to_16=False,
    ),
)

train_dataloader = dict(
    dataset=dict(band_names=rgb_band_names),
)
val_dataloader = dict(
    dataset=dict(band_names=rgb_band_names),
)
test_dataloader = dict(
    dataset=dict(band_names=rgb_band_names),
)
