_base_ = ['./dofav2-large_1xb4-50e_ningbo-rgb-frozen.py']

model = dict(
    backbone=dict(
        freeze_backbone=False,
        drop_path_rate=0.1,
    ),
)
optim_wrapper = dict(
    optimizer=dict(lr=6e-5),
)
