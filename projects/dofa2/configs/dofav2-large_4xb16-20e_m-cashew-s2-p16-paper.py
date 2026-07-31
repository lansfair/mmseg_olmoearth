_base_ = ['./dofav2-large_4xb16-20e_m-cashew-s2-paper.py']

# Controlled patch-size comparison: keep the paper protocol unchanged and
# convert only the pretrained 14x14 dynamic patch kernel to 16x16.
model = dict(
    backbone=dict(convert_patch_14_to_16=True),
)
