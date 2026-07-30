_base_ = [
    './_base_/models/dofav2_upernet.py',
    './_base_/datasets/svdt.py',
    './_base_/schedules/50e.py',
    '../../../configs/_base_/default_runtime.py',
]

model = dict(
    data_preprocessor=dict(
        mean=[72.4085, 89.7399, 69.6123],
        std=[32.8544, 23.9954, 23.1234],
        # MMSeg LoadImageFromFile decodes color images as BGR.
        bgr_to_rgb=True,
    ),
    backbone=dict(
        img_size=512,
        model_bands=['RED', 'GREEN', 'BLUE'],
    ),
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2),
)

auto_scale_lr = dict(enable=False, base_batch_size=4)
log_processor = dict(by_epoch=True)
