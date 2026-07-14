# UniverSat 在 PASTIS-R 上的评估

本项目评估 UniverSat 在 PASTIS-R 数据集上的语义分割性能。其结构遵循 `projects/copernicus/pastis` 和 `projects/olmoearth/pastis` 项目。

## 目录结构

```
pastis/
├── universat_pastis/           # PASTIS-R 组件的 Python 包
│   ├── datasets/pastis.py      # UniverSatPASTISDataset + 数据整理函数
│   ├── transforms/formatting.py
│   └── utils/norm.py
├── configs/
│   ├── universat-base_pastis_lp.py  # 线性探测（冻结骨干网络）
│   └── universat-base_pastis_ft.py  # 微调
├── train.sh
└── test.sh
```

骨干网络、解码头和数据预处理器复用自 `projects/universat/universat`。

## 数据布局

您的 PASTIS-R 目录应如下所示：

```
PASTIS-R/
  metadata.geojson              # ID_PATCH, Fold, dates-S2, dates-S1A
  DATA_S2/S2_{id}.npy           # T x 10 x H x W
  DATA_S1A/S1A_{id}.npy         # T x 3 x H x W
  ANNOTATIONS/TARGET_{id}.npy   # 1 x H x W 或 H x W
  NORM_S2_patch.json            # {"mean": [...], "std": [...]}
  NORM_S1_patch.json            # {"mean": [...], "std": [...]}
```

如果您没有归一化 JSON 文件，请先根据训练集拆分计算它们，或者在配置中临时设置 `norm_path=None`（不推荐用于实际评估）。

## 使用方法

在外部设置数据、权重和 GPU 环境变量；脚本会自动定位 MMSegmentation 根目录，并使用当前激活环境中的 Python（也可通过 `PYTHON` 指定）：

```bash
export MM_ARCHIVE_DATA_HOME=/path/to/data
export MM_ARCHIVE_CKPT_HOME=/path/to/checkpoints
bash projects/universat/pastis/train.sh
```

或者从 MMSegmentation 根目录手动运行：

```bash
export PYTHONPATH=".:$PWD/projects/universat:$PWD/projects/universat/pastis:$PYTHONPATH"
python tools/train.py \
    projects/universat/pastis/configs/universat-base_pastis_lp.py \
    --work-dir work_dirs/universat-base_pastis_lp
```

## 配置文件

- `universat-base_pastis_lp.py`：骨干网络冻结（`frozen_stages=0`），仅训练线性探测头。用于标准的 LP 评估。
- `universat-base_pastis_ft.py`：骨干网络解冻（`frozen_stages=-1`），使用小型卷积分割头对整个模型进行微调。

要切换到 UniverSat-Tiny，请将 `embed_dim` 改为 192，`num_heads` 改为 8，`block_type` 改为 `("Bi_ACA_in", "SAx12", "Bilinear_out", "CA_Sub")`（默认的 Base 配置已有 12 个 SA 块；Tiny 有 6 个）。

## 注意事项

- PASTIS-R 样本的时间序列长度可变。自定义整理函数会重复最后一个有效观测来补齐批次，避免把全零影像和参考日期误当成真实观测。默认批量大小为 1，以控制 128×128 稠密特征的显存占用。
- 传递给骨干网络的输入字典包含两个模态张量（`s2`、`s1`）及其对应的日期张量（`s2_dates`、`s1_dates`）。
- `output_grid=128` 表示骨干网络输出 128 x 128 的令牌网格，与 PASTIS-R 原生的 128 x 128 分辨率匹配。
- `num_classes=19` 且 `ignore_index=19`：模型预测 0=背景和 1-18 的有效类别；标注值 19 为空洞并被忽略。
