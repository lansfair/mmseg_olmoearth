# GeoFM Embedding：抽取、评测与推理

本项目位于完整的 `mmseg_olmoearth` 仓库中，负责：

- 从地球基础模型抽取全局或稠密 embedding；
- 将逐样本 PT/TIFF 汇总为统一的 `train.pt`、`valid.pt`、`test.pt`；
- 运行 Linear、kNN、K-means、DBSCAN 和余弦检索；
- 使用训练好的稠密 Linear probe 对无标签数据推理并输出原图大小 PNG。

当前适配器包括 OLMoEarth、官方 OLMoEarth wrapper、DINOv3、
CopernicusFM、TESSERA 和 UniverSAT。

## 1. 服务器环境与固定路径

以下环境和路径已在当前 `htzzb2` 服务器核对：

| 项目 | 服务器路径 |
| --- | --- |
| Conda 环境 | `/mnt/ht2-nas2/EO_test/miniconda3/envs/geofm-olmoearth-cu121` |
| 地球基础模型权重根目录 | `/mnt/ht2-nas2/EO_test/wyf/embedding_code/地球基础模型权重` |
| 数据集根目录 | `/mnt/ht2-nas2/EO_test/openmmlab-archive/dat` |
| PASTIS-R 原始数据 | `/mnt/ht2-nas2/EO_test/openmmlab-archive/dat/PASTIS-R` |
| PASTIS-R 处理后数据 | `/mnt/ht2-nas2/EO_test/openmmlab-archive/dat/PASTIS-R/dataset_for_OEF_64` |
| Potsdam | `/mnt/ht2-nas2/EO_test/openmmlab-archive/dat/potsdam` |
| 项目仓库 | `/mnt/ht2-nas2/EO_test/wyf/embedding_code/geofm_a100/src/mmseg_olmoearth` |

需要激活的环境是 `geofm-olmoearth-cu121`。打开一个新终端后，先执行：

```bash
conda activate geofm-olmoearth-cu121

export GEOF_REPO=/mnt/ht2-nas2/EO_test/wyf/embedding_code/geofm_a100/src/mmseg_olmoearth
export GEOF_WEIGHT_ROOT='/mnt/ht2-nas2/EO_test/wyf/embedding_code/地球基础模型权重'
export GEOF_MODEL_ROOT="$GEOF_WEIGHT_ROOT/geofm"
export GEOF_DATA_ROOT=/mnt/ht2-nas2/EO_test/openmmlab-archive/dat
export PASTIS_RAW_ROOT="$GEOF_DATA_ROOT/PASTIS-R"
export PASTIS_ROOT="$PASTIS_RAW_ROOT/dataset_for_OEF_64"
export PASTIS_DIR="$PASTIS_ROOT"
export POTSDAM_ROOT="$GEOF_DATA_ROOT/potsdam"
export GEOF_RESULT_ROOT=/mnt/ht2-nas2/EO_test/wyf/embedding_code/geofm_a100/results
export GEOF_EMBED_ROOT=$GEOF_RESULT_ROOT/embeddings
export GEOF_EVAL_ROOT=$GEOF_RESULT_ROOT/evaluation
export GEOF_PRED_ROOT=$GEOF_RESULT_ROOT/predictions
export POTSDAM_EXISTING_TIFF=/mnt/htzzb2/EO_test/cyz/Potsdam_embed_copfm

cd "$GEOF_REPO"
mkdir -p "$GEOF_EMBED_ROOT" "$GEOF_EVAL_ROOT" "$GEOF_PRED_ROOT"
```

可用下面的命令确认没有误用 `(base)` 环境：

```bash
which python
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`which python` 应输出：

```text
/mnt/ht2-nas2/EO_test/miniconda3/envs/geofm-olmoearth-cu121/bin/python
```

已核对的 OLMoEarth Base 权重为：

```text
/mnt/ht2-nas2/EO_test/wyf/embedding_code/地球基础模型权重/geofm/olmoearth/base/config.json
/mnt/ht2-nas2/EO_test/wyf/embedding_code/地球基础模型权重/geofm/olmoearth/base/weights.pth
```

PASTIS-R 的原始数据与处理后数据目录分别为：

```text
/mnt/ht2-nas2/EO_test/openmmlab-archive/dat/PASTIS-R
/mnt/ht2-nas2/EO_test/openmmlab-archive/dat/PASTIS-R/dataset_for_OEF_64
```

后文统一直接使用 `python`，因此不会误用服务器的 `(base)` 环境。模型配置中的
checkpoint 路径统一从 `$GEOF_MODEL_ROOT` 取，数据配置中的 `data_root` 统一指向
`$GEOF_DATA_ROOT` 下对应的数据集目录。

### 1.1 使用 OLMoEarth 官方脚本处理 PASTIS-R

PASTIS-R 不能直接把原始 `DATA_S1A`、`DATA_S1D`、`DATA_S2` 和标注目录交给
后续 embedding 脚本。应先使用 `olmoearth_pretrain` 自带的
`pastis_processor.py` 生成官方评测格式。该模块已经安装在
`geofm-olmoearth-cu121` 环境中，因此直接使用 `python`：

```bash
python -m olmoearth_pretrain.evals.datasets.pastis_processor \
  --data_dir "$PASTIS_RAW_ROOT" \
  --output_dir "$PASTIS_ROOT"
```

不添加 `--orig_size` 时，脚本将影像处理为当前项目使用的 64×64 版本。处理完成后
必须同时存在三个 split：

```text
$PASTIS_ROOT/
├── pastis_r_train/
│   ├── s1_images/
│   ├── s2_images/
│   ├── months.pt
│   └── targets.pt
├── pastis_r_valid/
│   ├── s1_images/
│   ├── s2_images/
│   ├── months.pt
│   └── targets.pt
└── pastis_r_test/
    ├── s1_images/
    ├── s2_images/
    ├── months.pt
    └── targets.pt
```

可用下面的命令检查处理是否完整：

```bash
for split in train valid test; do
  test -f "$PASTIS_ROOT/pastis_r_${split}/months.pt" && \
  test -f "$PASTIS_ROOT/pastis_r_${split}/targets.pt" && \
  test -d "$PASTIS_ROOT/pastis_r_${split}/s1_images" && \
  test -d "$PASTIS_ROOT/pastis_r_${split}/s2_images" && \
  echo "$split: OK" || echo "$split: INCOMPLETE"
done
```

后续 OLMoEarth 数据加载器的 `PASTIS_DIR` 应指向处理后根目录 `$PASTIS_ROOT`，
而不是原始目录 `$PASTIS_RAW_ROOT`。

## 2. 总体流程

```text
模型权重 + 原始数据 + MMSeg config
                │
                ▼
        extract_embeddings.py
                │
          逐样本 PT/TIFF + manifest
                │
                ▼
      pack_embedding_bundle.py
                │
       train.pt / valid.pt / test.pt
                │
       ┌────────┼──────────┬────────────┐
       ▼        ▼          ▼            ▼
    Linear     kNN      K-means/      余弦检索
                         DBSCAN
       │
       ▼
 Linear probe 权重
       │
       ▼
 predict_linear.py → 原图大小类别索引 PNG
```

如果 embedding 已经保存为 TIFF/PT，可跳过模型抽取，直接从 manifest 打包。

## 3. 统一 PT 目录和格式

目录结构固定为：

```text
$GEOF_EMBED_ROOT/<dataset>/<model>/
├── train.pt
├── valid.pt      # 可选
└── test.pt       # 可选，可以没有 labels
```

评测命令的前两个位置参数 `dataset`、`model` 必须和目录名一致。

| 任务 | embeddings | labels |
| --- | --- | --- |
| 场景分类 | `[N,D]` | `[N]` |
| 空间特征分类 | `[N,Hf,Wf,D]` | `[N]` |
| 稠密语义分割 | `[N,Hf,Wf,D]` | `[N,H,W]` |
| 无标签分割推理 | `[N,Hf,Wf,D]` | 不需要 |

bundle 还可包含：

- `sample_ids`：样本和输出文件名的对应关系；
- `source_shapes [N,2]`：每张原图的高、宽；
- `embedding_layout`：`ND`、`NHWD`，或兼容输入 `NDHW`。

## 4. 从模型抽取 embedding

下面以服务器上的 Potsdam 和 OLMoEarth Base 为例抽取训练集稠密特征：

```bash
python "$GEOF_REPO/projects/geofm_embeddings/tools/extract_embeddings.py" \
  "$GEOF_REPO/projects/geofm_embeddings/configs/olmoearth/olmoearth-base_potsdam-rgb.py" \
  "$GEOF_RESULT_ROOT/potsdam_export" \
  --split train \
  --mode dense \
  --dense-format pt \
  --save-labels \
  --cfg-options \
  "model.backbone.adapter.model_config_path=$GEOF_MODEL_ROOT/olmoearth/base/config.json" \
  "model.backbone.adapter.init_cfg.checkpoint=$GEOF_MODEL_ROOT/olmoearth/base/weights.pth" \
  "train_dataloader.dataset.data_root=$POTSDAM_ROOT" \
  train_dataloader.batch_size=4
```

使用规则：

- 分类使用 `--mode global`；分割使用 `--mode dense`。
- 稠密结果可用 `--dense-format pt` 或 `geotiff`。
- 有标签 split 添加 `--save-labels`。
- 无标签 split 不添加 `--save-labels`，其数据 pipeline 也不能包含
  `LoadAnnotations`。
- `--split train/val/test` 决定读取 config 中对应的 dataloader。
- 抽取其他模型时，换用相应 config，并把权重指向 `$GEOF_MODEL_ROOT` 下
  对应模型目录；`$GEOF_WEIGHT_ROOT` 保留为用户给出的完整权重根目录。

## 5. 从 manifest 汇总为 PT

### 5.1 有标签数据

```bash
mkdir -p "$GEOF_EMBED_ROOT/potsdam/olmoearth_base"

python "$GEOF_REPO/projects/geofm_embeddings/tools/pack_embedding_bundle.py" \
  "$GEOF_RESULT_ROOT/potsdam_export/train.json" \
  "$GEOF_EMBED_ROOT/potsdam/olmoearth_base/train.pt"
```

将 `train` 替换为 `valid` 或 `test`，即可生成对应 split。

### 5.2 无标签数据

```bash
python "$GEOF_REPO/projects/geofm_embeddings/tools/pack_embedding_bundle.py" \
  "$GEOF_RESULT_ROOT/potsdam_export/test.json" \
  "$GEOF_EMBED_ROOT/potsdam/olmoearth_base/test.pt" \
  --allow-unlabeled
```

### 5.3 已有 embedding TIFF

服务器已有 TIFF embedding：

```text
/mnt/htzzb2/EO_test/cyz/Potsdam_embed_copfm
```

该目录已经包含 `train.json`、`val.json`、逐样本 `embedding.tif` 和
`label.tif`，可直接打包：

```bash
mkdir -p "$GEOF_EMBED_ROOT/potsdam/copernicusfm"

python "$GEOF_REPO/projects/geofm_embeddings/tools/pack_embedding_bundle.py" \
  "$POTSDAM_EXISTING_TIFF/train.json" \
  "$GEOF_EMBED_ROOT/potsdam/copernicusfm/train.pt"

python "$GEOF_REPO/projects/geofm_embeddings/tools/pack_embedding_bundle.py" \
  "$POTSDAM_EXISTING_TIFF/val.json" \
  "$GEOF_EMBED_ROOT/potsdam/copernicusfm/valid.pt"
```

其他已有 TIFF 目录只需准备同样格式的 manifest。每个样本至少写明：

```json
{
  "sample_id": "tile_001",
  "embedding_path": "test/tile_001/embedding.tif",
  "seg_map_path": "test/tile_001/label.tif",
  "source_shape": [512, 512]
}
```

其中相对路径以 manifest 所在目录为基准。无标签数据省略 `seg_map_path`，
并按第 5.2 节在打包时添加 `--allow-unlabeled`。

打包器支持 embedding `.pt`/TIFF，并识别标签字段 `label_path` 和
`seg_map_path`。

## 6. Linear：有 valid 时自动选择学习率和 epoch

```bash
python "$GEOF_REPO/projects/geofm_embeddings/tools/evaluate_linear.py" \
  potsdam olmoearth_base \
  --embedding-root "$GEOF_EMBED_ROOT" \
  --output-root "$GEOF_EVAL_ROOT" \
  --split valid \
  --ignore-label 255 \
  --device cuda
```

Linear 会根据 PT 内容执行：

- `[N,D]+[N]`：场景分类 Linear probe；
- `[N,Hf,Wf,D]+[N,H,W]`：稠密语义分割 Linear probe。

每个学习率内部按 valid 指标选择 epoch，再跨学习率选择最终结果：分类使用
accuracy，分割使用 mIoU。最终权重位于：

```text
$GEOF_EVAL_ROOT/potsdam/olmoearth_base/linear/best_probe.pth
```

指定 `--split test` 时，仍使用有标签 valid 选参，最后在有标签 test 上报告。

## 7. Linear：只有 train 时仅训练

没有有标签 valid 时，只读取 `train.pt`：

```bash
python "$GEOF_REPO/projects/geofm_embeddings/tools/evaluate_linear.py" \
  potsdam olmoearth_base \
  --embedding-root "$GEOF_EMBED_ROOT" \
  --output-root "$GEOF_EVAL_ROOT" \
  --train-only \
  --learning-rate 0.001 \
  --epochs 50 \
  --ignore-label 255 \
  --device cuda
```

此模式不搜索最佳学习率或最佳 epoch；最后一轮权重保存为：

```text
$GEOF_EVAL_ROOT/potsdam/olmoearth_base/linear_train_only/linear_probe.pth
```

## 8. 无标签分割推理并输出 PNG

先按第 5.2 节生成不含 labels 的 PT，再运行：

```bash
python "$GEOF_REPO/projects/geofm_embeddings/tools/predict_linear.py" \
  "$GEOF_EVAL_ROOT/potsdam/olmoearth_base/linear_train_only/linear_probe.pth" \
  "$GEOF_EMBED_ROOT/potsdam/olmoearth_base/test.pt" \
  "$GEOF_PRED_ROOT/potsdam/olmoearth_base/test" \
  --device cuda
```

输出为：

```text
$GEOF_PRED_ROOT/potsdam/olmoearth_base/test/
├── <sample_id>.png
└── predictions.json
```

PNG 是单通道类别索引图。程序将 Linear logits 双线性插值到原图大小后再
argmax：

- PT 有 `source_shapes` 时自动逐图恢复尺寸；
- 旧 PT 没有尺寸元数据时，手动添加 `--output-size HEIGHT WIDTH`；
- PNG 不保存 CRS、仿射变换等地理信息。

## 9. kNN 分类

```bash
python "$GEOF_REPO/projects/geofm_embeddings/tools/evaluate_knn.py" \
  scene_classification olmoearth_base \
  --embedding-root "$GEOF_EMBED_ROOT" \
  --output-root "$GEOF_EVAL_ROOT" \
  --split test \
  --k 20 \
  --temperature 0.07 \
  --device cuda
```

- 只用于样本级标签 `[N]`，不做像素级分割。
- `train.pt` 固定作为 gallery，`--split valid/test` 指定有标签 query。
- 空间 embedding `[N,Hf,Wf,D]` 先做空间均值池化。
- 输出 accuracy、balanced accuracy、macro/weighted F1、precision、recall
  和混淆矩阵。

## 10. K-means

```bash
python "$GEOF_REPO/projects/geofm_embeddings/tools/evaluate_kmeans.py" \
  potsdam olmoearth_base \
  --embedding-root "$GEOF_EMBED_ROOT" \
  --output-root "$GEOF_EVAL_ROOT" \
  --split test \
  --per-class 256 \
  --n-init 20
```

- 特征先做 L2 归一化。
- 稠密数据按类别采样像素，并将标签像素中心映射到最近特征 token。
- 未指定 `--clusters` 时使用观测类别数。
- 输出 Hungarian accuracy、NMI、ARI、purity 和 cosine silhouette。

K-means 算法本身无监督，但当前脚本需要标签做均衡采样和质量评估。

## 11. DBSCAN

```bash
python "$GEOF_REPO/projects/geofm_embeddings/tools/evaluate_dbscan.py" \
  potsdam olmoearth_base \
  --embedding-root "$GEOF_EMBED_ROOT" \
  --output-root "$GEOF_EVAL_ROOT" \
  --split test \
  --per-class 256 \
  --min-samples 5 10 20 \
  --eps-multipliers 0.9 1.0 1.1
```

- 使用 L2 归一化特征和 cosine 距离。
- 根据 k-distance knee 估计 eps，再运行多组倍率。
- 输出聚类数、噪声比例、Hungarian accuracy、NMI、ARI、purity 和
  silhouette。

与 K-means 相同，当前脚本需要标签做均衡采样和指标计算。

## 12. 余弦语义检索

```bash
python "$GEOF_REPO/projects/geofm_embeddings/tools/evaluate_cosine_retrieval.py" \
  potsdam olmoearth_base \
  --embedding-root "$GEOF_EMBED_ROOT" \
  --output-root "$GEOF_EVAL_ROOT" \
  --gallery-split train \
  --query-split test \
  --k-values 1 5 10 20
```

- gallery 和 query 必须是不同 split。
- 样本或稠密像素特征先做类别均衡采样与 L2 归一化。
- 同类别视为相关结果。
- 输出 MRR、Hit Rate@K、Precision@K、Recall@K、mAP@K。
- `details.jsonl` 保存每个 query 的检索明细。

## 13. 缺少某些 split 时怎么运行

| 已有数据 | 使用方式 |
| --- | --- |
| 有标签 train、valid、test | 所有评测；Linear 用 valid 选参、test 报告 |
| 有标签 train、valid，无 test | 评测用 valid；检索用 train → valid |
| 只有有标签 train | Linear 使用 `--train-only`；聚类可在 train 上分析 |
| 有标签 train + 无标签 valid/test | 先 `--train-only`，再用 `predict_linear.py` |
| 只有无标签 embedding | 不能计算监督指标；可加载已有 Linear 权重推理 |

## 14. 输出位置

所有输出均放在服务器目录：

```text
/mnt/ht2-nas2/EO_test/wyf/embedding_code/geofm_a100/results/
├── embeddings/     # train.pt / valid.pt / test.pt
├── evaluation/     # report.json、Linear 权重和检索明细
└── predictions/    # 无标签分割 PNG
```

评测报告的通用位置为：

```text
$GEOF_EVAL_ROOT/<dataset>/<model>/<task>/report.json
```

所有评测固定使用随机种子 `42`。比较不同模型时，必须保持数据划分、采样量、
Linear 超参数和 ignore label 一致。
