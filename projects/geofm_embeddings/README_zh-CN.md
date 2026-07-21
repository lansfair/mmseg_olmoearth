# GeoFM Embedding 工具

本项目位于完整的 `olmoearth_mmseg` 仓库中，用于统一抽取地球基础模型的
embedding，并对 `.pt` embedding 运行独立评测。

## 1. 目录约定

```text
<embedding-root>/<dataset>/<model>/
├── train.pt
├── valid.pt
└── test.pt          # 可选
```

每个文件是一个字典：

```python
{"embeddings": embeddings, "labels": labels}
```

支持三种组合：

- 分类：`embeddings [N,D]`，`labels [N]`
- 空间特征分类：`embeddings [N,Hf,Wf,D]`，`labels [N]`
- 稠密语义分割：`embeddings [N,Hf,Wf,D]`，`labels [N,H,W]`

## 2. 从模型抽取 embedding

Potsdam 示例配置：

```bash
python projects/geofm_embeddings/tools/extract_embeddings.py \
  projects/geofm_embeddings/configs/olmoearth/olmoearth-base_potsdam-rgb.py \
  work_dirs/potsdam_embeddings \
  --split train \
  --mode dense \
  --dense-format pt \
  --save-labels
```

分别把 `--split` 改为 `train`、`val`、`test`。数据路径和模型权重可用
`--cfg-options` 覆盖。分类任务使用 `--mode global`；分割任务使用
`--mode dense`。

当前模型适配器包括 OLMoEarth、官方 OLMoEarth wrapper、DINOv3、
CopernicusFM、TESSERA 和 UniverSAT。

## 3. 将逐样本文件汇总为 PT

```bash
python projects/geofm_embeddings/tools/pack_embedding_bundle.py \
  work_dirs/potsdam_embeddings/train.json \
  embeddings/potsdam/olmoearth_base/train.pt
```

验证集通常输出为 `val.json`，汇总时保存成 `valid.pt`。打包器同时识别
manifest 中的 `label_path` 和 `seg_map_path`，因此已有的 embedding TIFF 与
标签 TIFF 也可以直接汇总，不需要重新抽取。

## 4. 运行评测

### Linear

```bash
python projects/geofm_embeddings/tools/evaluate_linear.py \
  potsdam olmoearth_base \
  --embedding-root embeddings \
  --output-root work_dirs/geofm_eval \
  --split test \
  --ignore-label 255
```

该入口对 `[N,D]+[N]` 做分类 linear probe，对
`[N,Hf,Wf,D]+[N,H,W]` 做稠密语义分割 linear probe。
训练完成后，最佳稠密 Linear 权重保存在对应结果目录的
`best_probe.pth`。

没有有标签 valid 时，可只用 `train.pt` 训练。此模式不选择学习率或
epoch，必须明确指定一个学习率，最终保存最后一轮权重：

```bash
python projects/geofm_embeddings/tools/evaluate_linear.py \
  potsdam olmoearth_base \
  --embedding-root embeddings \
  --output-root work_dirs/geofm_eval \
  --train-only \
  --learning-rate 0.001 \
  --epochs 50 \
  --ignore-label 255 \
  --device cuda
```

权重路径为：

```text
work_dirs/geofm_eval/potsdam/olmoearth_base/linear_train_only/linear_probe.pth
```

### 无标签数据 Linear 推理并保存 PNG

无标签 split 抽取 embedding 时不要传 `--save-labels`，然后打包：

```bash
python projects/geofm_embeddings/tools/pack_embedding_bundle.py \
  work_dirs/potsdam_embeddings/test.json \
  embeddings/potsdam/olmoearth_base/test.pt \
  --allow-unlabeled
```

使用已经训练好的稠密 Linear probe 推理：

```bash
python projects/geofm_embeddings/tools/predict_linear.py \
  work_dirs/geofm_eval/potsdam/olmoearth_base/linear_train_only/linear_probe.pth \
  embeddings/potsdam/olmoearth_base/test.pt \
  work_dirs/potsdam_predictions \
  --device cuda
```

输出是单通道类别索引 PNG，并生成 `predictions.json`。如果 PT 中包含
`source_shapes`，PNG 会自动恢复到源图大小；旧 PT 没有该字段时可加
`--output-size HEIGHT WIDTH`。

### kNN

```bash
python projects/geofm_embeddings/tools/evaluate_knn.py \
  scene_classification olmoearth_base \
  --embedding-root embeddings \
  --output-root work_dirs/geofm_eval \
  --split test
```

kNN 只做样本级分类；`train.pt` 是 gallery，`--split` 指定 query。

### K-means、DBSCAN、余弦检索

```bash
python projects/geofm_embeddings/tools/evaluate_kmeans.py \
  potsdam olmoearth_base --embedding-root embeddings \
  --output-root work_dirs/geofm_eval --split test

python projects/geofm_embeddings/tools/evaluate_dbscan.py \
  potsdam olmoearth_base --embedding-root embeddings \
  --output-root work_dirs/geofm_eval --split test

python projects/geofm_embeddings/tools/evaluate_cosine_retrieval.py \
  potsdam olmoearth_base --embedding-root embeddings \
  --output-root work_dirs/geofm_eval \
  --gallery-split train --query-split test
```

K-means、DBSCAN 和余弦检索既可处理样本向量，也可按类别采样稠密像素
embedding。

## 5. 没有 test.pt

只有 `train.pt` 和 `valid.pt` 时：

- linear、kNN、K-means、DBSCAN 使用 `--split valid`
- 余弦检索使用 `--gallery-split train --query-split valid`

结果统一写入：

```text
<output-root>/<dataset>/<model>/<task>/report.json
```

所有评测固定使用一个随机种子 `42`。
