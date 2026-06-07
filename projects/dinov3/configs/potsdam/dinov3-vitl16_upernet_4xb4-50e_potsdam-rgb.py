_base_ = "./dinov3-vitl16_4xb4-50e_potsdam-rgb.py"

work_dir = "./work_dirs/dinov3-vitl16_upernet_4xb4-50e_potsdam-rgb-rvsa"

ignore_index = 5
num_classes = 5
hidden_dim = 1024
norm_cfg = dict(type="SyncBN", requires_grad=True)

train_dataloader = dict(
    dataset=dict(label_mapping="official_to_rvsa_class5_ignore5"),
)

val_dataloader = dict(
    dataset=dict(label_mapping="official_to_rvsa_class5_ignore5"),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="OlmoEarthIoUMetric",
    num_classes=num_classes,
    ignore_index=ignore_index,
    use_valid_mask=False,
)
test_evaluator = val_evaluator

model = dict(
    data_preprocessor=dict(seg_pad_val=ignore_index),
    backbone=dict(
        out_indices=(7, 11, 15, 23),
    ),
    neck=dict(
        type="MultiLevelNeck",
        in_channels=[hidden_dim, hidden_dim, hidden_dim, hidden_dim],
        out_channels=hidden_dim,
        scales=[4, 2, 1, 0.5],
        norm_cfg=norm_cfg,
    ),
    decode_head=dict(
        _delete_=True,
        type="UPerHead",
        in_channels=[hidden_dim, hidden_dim, hidden_dim, hidden_dim],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        ignore_index=ignore_index,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1.0,
        ),
    ),
    auxiliary_head=dict(
        type="FCNHead",
        in_channels=hidden_dim,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        ignore_index=ignore_index,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=0.4,
        ),
    ),
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.001, weight_decay=0.01),
)
