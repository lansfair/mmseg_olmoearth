_base_ = ['./dofav2-large_2xb32-30e_m-cashew-rgb-p14-paper.py']

model = dict(
    backbone=dict(convert_patch_14_to_16=True),
)
