# UniverSat 迁移至 MMSegmentation 1.x

本项目将 UniverSat 多模态地球观测编码器迁移到
[MMSegmentation 1.x](https://github.com/open-mmlab/mmsegmentation)，
作为 `mmsegmentation/projects/universat/` 下的外部项目，不侵入 MMSegmentation
源码本身。

> 本文档是 `README.md` 的中文版本。若两者有冲突，以最新修改的版本为准。

## 目录结构

```
projects/universat/
├── universat/                           # Python 包
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbones/
│   │   │   ├── __init__.py
│   │   │   ├── universat_backbone.py    # MMSeg 1.x backbone 包装器
│   │   │   └── universat_modules/       # 原始编码器核心代码
│   │   │       ├── UniverSat.py
│   │   │       ├── UniversalPatchEncoder.py
│   │   │       ├── modality_registry.py
│   │   │       ├── masking/
│   │   │       └── utils/
│   │   ├── decode_heads/
│   │   │   ├── universat_seg_head.py    # 分割头
│   │   │   └── universat_lp_head.py     # Linear probe 头
│   │   └── data_preprocessors.py        # 多模态数据预处理器
│   └── datasets/
│       ├── __init__.py
│       ├── universat_dataset.py         # 多模态数据集
│       └── transforms.py                # 多模态加载/打包变换
├── configs/
│   └── base_universat_seg.py            # 通用 JSON 数据集配置模板
├── pastis/                              # 独立的 PASTIS-R 项目
│   ├── universat_pastis/
│   ├── configs/
│   │   ├── universat-base_pastis_lp.py  # PASTIS-R linear probe
│   │   └── universat-base_pastis_ft.py  # PASTIS-R 微调
│   ├── train.sh
│   └── test.sh
├── train.sh                             # 训练启动脚本
├── test.sh                              # 测试启动脚本
└── README.md                            # 英文说明
```

## 环境要求

- MMSegmentation 1.x / OpenMMLab 2.0（使用 `mmengine` + `mmseg.registry`）
- PyTorch >= 2.0（默认使用稳定的 eager 模式，可通过 `compile_encoder=True` 启用动态编译）
- `safetensors`（用于加载官方 `.safetensors` 权重）
- `einops`（`flexiVit.py` 依赖）

```bash
pip install -r projects/universat/requirements.txt
```

## PASTIS-R 下游任务测评

如果需要在 PASTIS-R 上进行 linear probe / fine-tuning 的完整项目，
请参考 `pastis/` 子目录。它复用了 `projects/universat/universat` 中的
backbone、decode head 和 data preprocessor，并提供了专门处理 PASTIS-R
可变长度时间序列的数据集类、collate 函数和配置。

## 快速开始

### 1. 准备数据

为数据集制作 JSON 格式的 split 文件，例如：

```json
[
  {
    "filenames": {
      "s2": "s2/xxx.npy",
      "s1": "s1/xxx.npy"
    },
    "dates": {
      "s2": [365, 377, 389],
      "s1": [366, 378, 390]
    },
    "ann": {"seg_map": "masks/xxx.png"},
    "height": 360,
    "width": 360
  }
]
```

然后复制并修改 `configs/base_universat_seg.py`，
填入 `data_root`、`split` 路径、`num_classes`、`ignore_index` 以及各 modality
的 `mean`/`std` 归一化统计量。

### 2. 准备预训练权重

将 UniverSat 预训练权重（`.safetensors` 或 `.pth`）放到
`MM_ARCHIVE_CKPT_HOME` 指向的目录，并确保配置中的
`backbone.init_cfg.checkpoint` 路径正确。

### 3. 训练

在 MMSegmentation 根目录（包含 `tools/` 的目录）执行：

```bash
export PYTHONPATH=".:$PYTHONPATH"
export MM_ARCHIVE_DATA_HOME=/path/to/data
export MM_ARCHIVE_CKPT_HOME=/path/to/checkpoints
bash projects/universat/train.sh \
    projects/universat/pastis/configs/universat-base_pastis_lp.py
```

或直接使用提供的启动脚本：

```bash
cd projects/universat
bash train.sh
```

### 4. 测试

```bash
bash projects/universat/test.sh \
    projects/universat/pastis/configs/universat-base_pastis_lp.py \
    path/to/checkpoint.pth
```

首次训练前可检查一个样本及最小前向：

```bash
python projects/universat/tools/check_runtime.py \
    projects/universat/pastis/configs/universat-base_pastis_lp.py \
    --device cuda:0
```

## 核心组件说明

| 组件                          | 注册名         | 作用                                               |
| ----------------------------- | -------------- | -------------------------------------------------- |
| `UniverSatBackbone`         | `MODELS`     | 包装原始 UniverSat 编码器，输出 MMSeg 风格的特征图 |
| `UniverSatSegHead`          | `MODELS`     | 轻量卷积分割头                                     |
| `UniverSatLinearProbeHead`  | `MODELS`     | LayerNorm + 1×1 分类器，用于 linear probe         |
| `UniverSatDataPreprocessor` | `MODELS`     | 将多模态 dict 透传给 backbone                      |
| `UniverSatSegDataset`       | `DATASETS`   | 读取 JSON split 的多模态分割数据集                 |
| `LoadMultimodalFromFile`    | `TRANSFORMS` | 加载各 modality 的`.npy` 数据                    |
| `NormalizeMultimodal`       | `TRANSFORMS` | 按 modality 独立归一化                             |
| `PackUniverSatInputs`       | `TRANSFORMS` | 打包多模态输入与标注为`SegDataSample`            |

## 适配自己的数据集

1. **修改 modality 列表**：在数据集配置和 backbone 配置中同步更新 `modalities`。
2. **补充 modality 元数据**：如果某个 modality 不在 `modality_registry.py` 中，
   在 backbone 参数里提供 `wavelengths`、`input_res`、`subpatches` 覆盖。
3. **替换归一化统计量**：将 `mean`/`std` 替换为训练集上计算的真实值。
4. **对齐空间尺寸**：确保 `patch_size`、`output_grid`、`crop_size` 与输入 patch
   布局一致。`output_grid` 表示方形网格的边长，例如 `36` 表示 36×36 个 token。

## 常见问题

### 权重加载时出现 missing / unexpected keys

- 官方权重通常带有 `model.` 前缀，包装器会自动剥离。
- SSL 预训练权重中的 `projector__*` 头会被自动忽略，这是预期行为。
- 如果 missing keys 包含 backbone 核心参数，请检查下载的权重版本与
  `block_type`、`embed_dim`、`n_registers` 等配置是否匹配。

### 输入不是 dict

`UniverSatBackbone` 要求输入为 `{modality: tensor}` 字典。
请确保数据集返回的 `img` 是 dict，并且 `UniverSatDataPreprocessor`
被配置为 `model.data_preprocessor`。

空间快照使用 `(B,C,H,W)`；时间序列使用 `(B,T,C,H,W)`，并必须同时提供
`<modality>_dates`，形状为 `(B,T)`。项目会在进入编码器前检查这些维度。

### 出现 `KeyError: modality not in modality_registry`

为该 modality 提供显式参数：

```python
backbone=dict(
    type='UniverSatBackbone',
    modalities=['my_mod'],
    wavelengths={'my_mod': [0.49, 0.56, 0.665]},
    input_res={'my_mod': 10.0},
    subpatches={'my_mod': 1},
    ...
)
```
