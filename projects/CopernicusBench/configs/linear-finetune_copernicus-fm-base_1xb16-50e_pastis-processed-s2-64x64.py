_base_ = ['./linear-probe_copernicus-fm-base_1xb16-50e_pastis-processed-s2-64x64.py']

model = dict(
    backbone=dict(
        frozen_exclude=['all'],
        norm_eval=False,
    ))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.01),
)
param_scheduler = [
    dict(
        type='OneCycleLR',
        eta_max=1e-4,
        pct_start=0.1,
        anneal_strategy='cos',
        begin=0,
        end=50,
        by_epoch=True,
        convert_to_iter_based=True,
    )
]
