_base_ = ['./dofav2-large_1xb12-20e_m-cashew-s2-frozen.py']

model = dict(
    backbone=dict(
        freeze_backbone=False,
        drop_path_rate=0.1,
    ),
)
optim_wrapper = dict(
    optimizer=dict(lr=6e-5),
)
