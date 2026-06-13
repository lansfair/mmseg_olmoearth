_base_ = "./olmoearth-base_1xb8-50e_crop-type-s2-linear.py"

work_dir = "./work_dirs/olmoearth-base_1xb8-50e_crop-type-s2-ft"

olmoearth_model_dir = "/mnt/ht2-nas2/EO_test/model/OlmoEarth-v1-Base"
model_config_path = f"{olmoearth_model_dir}/config.json"
weights_path = f"{olmoearth_model_dir}/weights.pth"

# Align olmoearth_pretrain finetune eval for m-SA-crop-type:
# ft_batch_size=8, num_workers=2, epochs=50, patch_size=4,
# NORM_NO_CLIP_2_STD, and 20% frozen-backbone warm start.
custom_hooks = [
    dict(
        type="FreezeBackboneUntilEpochHook",
        unfreeze_epoch=10,
    )
]

model = dict(
    backbone=dict(
        model_config_path=model_config_path,
        init_cfg=dict(type="Pretrained", checkpoint=weights_path),
        modality="sentinel2_l2a",
        fast_pass=True,
    ),
)

optim_wrapper = dict(
    type="AmpOptimWrapper",
    loss_scale="dynamic",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(
        type="MultiStepLR",
        begin=0,
        end=50,
        milestones=[10],
        gamma=0.1,
        by_epoch=True,
    ),
    dict(
        type="ReduceOnPlateauParamScheduler",
        param_name="lr",
        monitor="mIoU",
        rule="greater",
        factor=0.2,
        patience=2,
        min_value=0.0,
        cooldown=10,
        by_epoch=True,
    ),
]

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=50, val_interval=10)

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=True,
        interval=10,
        save_best="mIoU",
        rule="greater",
        max_keep_ckpts=3,
    ),
)
