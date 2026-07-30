_base_ = [
    './_base_/models/dofav2_upernet.py',
    './_base_/datasets/m-cashew-plant.py',
    './_base_/schedules/20e.py',
    '../../../configs/_base_/default_runtime.py',
]

# One GPU uses 12 samples. Enable auto_scale_lr explicitly when changing the
# total batch size; the reference learning rate below is based on 12.
auto_scale_lr = dict(enable=False, base_batch_size=12)
log_processor = dict(by_epoch=True)
