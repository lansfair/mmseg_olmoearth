optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=0.05),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        begin=0,
        end=3,
        convert_to_iter_based=True,
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-6,
        by_epoch=True,
        begin=3,
        end=20,
        T_max=17,
        convert_to_iter_based=True,
    ),
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=20,
    val_interval=1,
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(
        type='LoggerHook',
        interval=50,
        log_metric_by_epoch=True,
    ),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=1,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=3,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'),
)
