_base_ = ['./dofav2-large_4xb16-20e_m-cashew-rgb-paper.py']

model = dict(
    backbone=dict(convert_patch_14_to_16=True),
)
