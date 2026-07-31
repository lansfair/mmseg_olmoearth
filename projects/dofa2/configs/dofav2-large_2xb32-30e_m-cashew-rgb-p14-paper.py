_base_ = ['./dofav2-large_4xb16-20e_m-cashew-rgb-paper.py']

# Two GPUs x 32 samples keeps the paper's global batch size of 64.
train_dataloader = dict(batch_size=32)
val_dataloader = dict(batch_size=32)
test_dataloader = dict(batch_size=32)

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=0,
        end=30,
        T_max=30,
        eta_min=0.0,
    ),
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=30,
    val_interval=1,
)
