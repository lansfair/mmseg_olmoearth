# 遥感大图滑窗推理脚本

这个 project 提供一个统一的大图 GeoTIFF 推理入口，用于
MMSegmentation 语义分割模型的遥感大图滑窗推理。

统一入口：

推荐把核心脚本当作底层入口，把常用模型/数据集组合写成独立场景脚本。
例如 OLMoEarth 在 Potsdam RGB 大图上的推理可以直接使用：

```bash
python projects/rs_large_infer/tools/infer_olmoearth_potsdam_rgb.py \
  --image /data/potsdam_large_rgb.tif \
  --checkpoint /checkpoints/olmoearth_potsdam.pth \
  --output /outputs/potsdam_pred.tif
```

DINOv3 在 Potsdam RGB 大图上的推理可以使用：

```bash
python projects/rs_large_infer/tools/infer_dinov3_potsdam_rgb.py \
  --image /data/potsdam_large_rgb.tif \
  --checkpoint /checkpoints/dinov3_potsdam.pth \
  --output /outputs/potsdam_pred.tif
```

也可以先编辑这个文件顶部的 `IMAGE`、`CHECKPOINT`、`OUTPUT` 等变量，
然后不带参数运行。没有传入的参数会读取脚本顶部默认值，传入的命令行参数
会覆盖同名默认值。

通用入口仍然是：

```bash
python projects/rs_large_infer/src/cli.py \
  --image /path/to/large.tif \
  --config /path/to/config.py \
  --checkpoint /path/to/checkpoint.pth \
  --output /path/to/pred_label.tif \
  --window-size 512 512 \
  --stride 256 256 \
  --batch-size 1 \
  --device cuda:0
```

脚本会做这些事情：

- 使用 `rasterio` 读取大 GeoTIFF。
- 按 `--window-size` 和 `--stride` 做滑窗。
- 对每个窗口构造 MMSeg `inputs` 和 `SegDataSample`。
- 调用 `model.test_step` 批量推理。
- 将预测类别图写回单波段 GeoTIFF，并保留原图的空间参考信息。

## 输入模式

`--input-mode` 支持：

```text
auto        自动识别 CopernicusFM、OLMoEarth RGB/S2，否则走 standard。
standard    普通 RGB 或多波段 MMSeg 模型，例如 DINOv3、SegFormer、UPerNet。
rgb         OLMoEarth RGB 适配模式，使用 RGBToOlmoEarthS2。
s2          OLMoEarth Sentinel-2 模式，处理 S2 波段顺序和 OLMoEarth metadata。
copernicus  CopernicusFM 模式，生成 [lon, lat, time, patch_area] metadata。
```

通常建议使用默认的 `auto`。只有自动识别不符合预期时，再手动指定
`standard`、`rgb`、`s2` 或 `copernicus`。

## 普通模型示例

适用于 DINOv3、SegFormer、UPerNet 等不需要特殊遥感 metadata 的模型：

```bash
python projects/rs_large_infer/src/cli.py \
  --image /data/large_rgb.tif \
  --config projects/dinov3/configs/potsdam/dinov3-vitl16_4xb4-50e_potsdam-rgb.py \
  --checkpoint /checkpoints/dinov3_mmseg.pth \
  --output /outputs/pred.tif \
  --input-mode standard \
  --window-size 512 512 \
  --stride 256 256 \
  --batch-size 2 \
  --device cuda:0
```

如果普通多波段模型只需要读取部分 band，可以使用 1-based band 编号：

```bash
--band-indices 1 2 3
```

如果需要对读取的 band 做缩放：

```bash
--band-scales 0.0001 0.0001 0.0001
```

## OLMoEarth RGB 示例

RGB GeoTIFF 会按文件 band 顺序读取，通常是 R/G/B：

```bash
python projects/rs_large_infer/src/cli.py \
  --image /data/rgb_large.tif \
  --config projects/olmoearth/configs/potsdam/olmoearth-base_upernet_4xb4-80k_potsdam-rgb-p4-512x512.py \
  --checkpoint /checkpoints/mmseg_checkpoint.pth \
  --output /outputs/pred_label.tif \
  --input-mode rgb \
  --window-size 512 512 \
  --stride 256 256 \
  --batch-size 1 \
  --device cuda:0 \
  --rgb-channel-order RGB \
  --cfg-options \
  model.backbone.model_config_path=/checkpoints/olmoearth/config.json \
  model.backbone.init_cfg.checkpoint=/checkpoints/olmoearth/weights.pth
```

只有当输入 GeoTIFF 真的是 B/G/R 存储时，才需要：

```bash
--rgb-channel-order BGR
```

## OLMoEarth Sentinel-2 示例

```bash
python projects/rs_large_infer/src/cli.py \
  --image /data/s2_large.tif \
  --config projects/olmoearth/configs/dfc2020_s2/olmoearth-base_4xb4-50e_dfc2020-s2.py \
  --checkpoint /checkpoints/mmseg_checkpoint.pth \
  --output /outputs/pred_label.tif \
  --input-mode s2 \
  --window-size 256 256 \
  --stride 128 128 \
  --batch-size 1 \
  --device cuda:0 \
  --source-band-names B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B10 B11 B12 \
  --cfg-options \
  model.backbone.model_config_path=/checkpoints/olmoearth/config.json \
  model.backbone.init_cfg.checkpoint=/checkpoints/olmoearth/weights.pth
```

OLMoEarth S2 模式最终会重排到下面 12 个 band 顺序：

```text
B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09
```

如果不传 `--source-band-names`，输入 GeoTIFF 必须已经是这个 12-band 顺序。

## CopernicusFM 示例

CopernicusFM 需要 `copernicus_meta`：

```text
[lon, lat, time, patch_area]
```

脚本会按每个滑窗中心点计算经纬度，并从 config 或命令行得到时间和
`patch_area`。

```bash
python projects/rs_large_infer/src/cli.py \
  --image /data/large_s2.tif \
  --config projects/CopernicusBench/configs/upernet_copernicus-fm-base_1xb16-50e_dfc2020-s2-256x256.py \
  --checkpoint /checkpoints/mmseg_checkpoint.pth \
  --output /outputs/pred_label.tif \
  --input-mode copernicus \
  --window-size 256 256 \
  --stride 128 128 \
  --batch-size 1 \
  --device cuda:0
```

如果大图文件名不包含可解析日期，可以手动指定：

```bash
--copernicus-date 20240501
```

或者直接指定 days since 1970-01-01：

```bash
--copernicus-date-days 19844
```

如果影像缺少 CRS，或者你想固定所有窗口的经纬度：

```bash
--copernicus-lon-lat 116.3 39.9
```

如果需要覆盖 `patch_area`：

```bash
--copernicus-patch-area 0.0256
```

## 常用参数

```text
--window-size W H       滑窗大小，单位像素。
--stride X Y            滑窗步长，单位像素。小于 window-size 时会有重叠。
--batch-size N          每次 forward 的窗口数量。
--device cuda:0         推理设备。
--cfg-options           覆盖 config 中的字段，语法同 tools/test.py。
```

输出是单波段类别 GeoTIFF。类别数不超过 255 时写 `uint8`，不超过 32767
时写 `int16`，更大时写 `int32`。



