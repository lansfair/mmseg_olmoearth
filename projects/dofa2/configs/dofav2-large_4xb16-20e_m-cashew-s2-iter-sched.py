_base_ = ['./dofav2-large_4xb16-20e_m-cashew-s2-official.py']

# Match the official source's per-iteration learning-rate update. This config
# intentionally changes only scheduler granularity relative to ``official`` so
# the effect can be measured independently.
param_scheduler = [
    dict(
        type='LinearLR',
        by_epoch=True,
        begin=0,
        end=3,
        start_factor=1e-3,
        convert_to_iter_based=True,
    ),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=3,
        end=20,
        T_max=17,
        eta_min=1e-6,
        convert_to_iter_based=True,
    ),
]
