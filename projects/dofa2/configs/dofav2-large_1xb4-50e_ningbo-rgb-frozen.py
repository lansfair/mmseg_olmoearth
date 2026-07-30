_base_ = [
    './_base_/models/dofav2_upernet.py',
    './_base_/datasets/ningbo-2m.py',
    './_base_/schedules/50e.py',
    '../../../configs/_base_/default_runtime.py',
]

model = dict(
    data_preprocessor=dict(
        mean=[123.675, 116.280, 103.530],
        std=[58.395, 57.120, 57.375],
        # Rasterio preserves the file's R/G/B band order.
        bgr_to_rgb=False,
    ),
    backbone=dict(
        img_size=512,
        model_bands=['RED', 'GREEN', 'BLUE'],
    ),
    decode_head=dict(num_classes=8),
    auxiliary_head=dict(num_classes=8),
)

auto_scale_lr = dict(enable=False, base_batch_size=4)
log_processor = dict(by_epoch=True)
