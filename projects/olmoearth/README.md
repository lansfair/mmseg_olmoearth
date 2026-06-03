# OLMoEarth for MMSegmentation

This project is a non-invasive OpenMMLab migration for OLMoEarth downstream
semantic segmentation tasks. It is organized around explicit converted data
manifests instead of wrapping `rslearn.train.dataset.ModelDataset` during
training.

## Which OpenMMLab Project

Use the OLMoEarth project that matches the downstream task:

| Task | Project | Typical data |
| --- | --- | --- |
| Semantic segmentation | MMSegmentation | masks, valid masks, GeoTIFF manifests |
| Horizontal-box detection | MMDetection | rslearn detection manifest, VOC/XML DIOR |
| Oriented-box detection | MMRotate | DOTA txt, DIOR-R oriented XML |

## Design

- `OlmoEarthSegDataset` is a manifest-backed `BaseDataset`, so it keeps
  OpenMMLab dataset lifecycle behavior while avoiding training-time rslearn
  wrapping.
- `LoadOlmoEarthArrays` loads imagery, labels, timestamps, and optional valid
  masks.
- `OlmoEarthSegDataPreProcessor` extends MMSeg padding so optional valid masks
  stay aligned with padded inputs and labels.
- `OlmoEarthBackbone` builds the OLMoEarth encoder from `model_config_path`,
  loads released weights through OpenMMLab `init_cfg`, and reuses OLMoEarth's
  reference `pool_unmasked_tokens` path to produce dense features.
- `OlmoEarthFeatureBackbone` is the offline-probe path: it reads precomputed
  dense OLMoEarth embeddings as the model input and trains only the probe head,
  matching OLMoEarth's original linear-probe evaluation shape.
- `OlmoEarthEncoderDecoder` passes temporal metadata from `SegDataSample` to the
  backbone.
- `OlmoEarthPatchLinearHead` implements the paper-style patch-linear dense
  probe. `OlmoEarthLinearHead` remains available for conventional MMSeg-style
  upsample-and-classify experiments.
- `OlmoEarthIoUMetric` keeps OLMoEarth's optional valid-mask filtering while
  reporting MMSeg-style `aAcc`, `mIoU`, `mAcc`, and optional F-score metrics.
- `OlmoEarthVisualizationHook` avoids MMSeg's default RGB-file assumption and
  renders validation/test overlays from the actual multiband batch tensor.
- `OlmoEarthPad` and `OlmoEarthCrop` reproduce the rslearn pad/crop step used
  by AWF and Nandi before normalizing Sentinel-2 inputs.

The primary dependency path is the local/full `olmoearth_pretrain` package.
Converted OLMoEarth manifest datasets are GeoTIFF-only, so both conversion and
training require `rasterio`. The runtime does not read paths from environment
variables; set `data_root`, `model.backbone.model_config_path`, and
`model.backbone.init_cfg.checkpoint` directly in config files or with
`--cfg-options`.

Conversion and embedding tools show progress. If `tqdm` is installed they use a
progress bar; otherwise they fall back to periodic `current/total` prints.

## Checkpoint Layout

The configs expect the released OLMoEarth files to be laid out as:

```text
checkpoints/olmoearth/
  config.json
  weights.pth
```

`model_config_path` is used only to build the OLMoEarth model structure.
`init_cfg.checkpoint` is used only for loading the released OLMoEarth
`weights.pth`. MMSegmentation's top-level `load_from` should still be reserved
for resuming or initializing a full MMSeg checkpoint.

## Data Layout

Converted data should look like:

```text
data/olmoearth_mmseg/pastis/
  train.json
  val.json
  test.json
  metainfo.json
  samples/train_000000/
    t00_sentinel2_l2a.tif
    t01_sentinel2_l2a.tif
    ...
    label.tif
    valid_mask.tif
```

Image arrays are stored as one raw, unnormalized GeoTIFF per timestep.
Normalization is done in the MMSeg pipeline with OLMoEarth computed
statistics. Converters also write `metainfo.json` with class counts, band
order, split label statistics, and normalization provenance so converted data
can be audited before training.

Manifests use `img_paths`, with one GeoTIFF per timestep:

```json
{
  "sample_id": "train_000000",
  "img_paths": [
    "samples/train_000000/t00_sentinel2_l2a.tif",
    "samples/train_000000/t01_sentinel2_l2a.tif"
  ],
  "seg_map_path": "samples/train_000000/label.tif",
  "timestamps": [[1, 4, 2020], [1, 5, 2020]],
  "olmoearth_modality": "sentinel2_l2a",
  "olmoearth_num_timesteps": 2
}
```

Each image GeoTIFF is read as `[C, H, W]`, the list is stacked as
`[T, C, H, W]`, and the existing OLMoEarth loader flattens it to MMSeg's
channel-first tensor. Labels and valid masks are GeoTIFFs too.

## PASTIS

Use the processed PASTIS-R tensors created by OLMoEarth:

```bash
python projects/olmoearth/tools/convert_pastis.py \
  --input-root /path/to/pastis_r \
  --output-root data/olmoearth_mmseg/pastis
```

Then train:

```bash
python tools/train.py \
  projects/olmoearth/configs/pastis/olmoearth-base_4xb4-50e_pastis-s2.py \
  --cfg-options \
  model.backbone.model_config_path=/path/to/olmoearth/config.json \
  model.backbone.init_cfg.checkpoint=/path/to/olmoearth/weights.pth
```

When overriding paths from the command line, set the nested config keys. The
top-level `data_root`, `model_config_path`, and `weights_path` variables are
readability helpers inside the config file; overriding them after parsing does
not rewrite the already-expanded nested dictionaries.

PASTIS void label `19` is converted by OLMoEarth preprocessing to `-1`; the
converter maps ignored pixels to MMSeg `ignore_index=255`.

The provided PASTIS, MADOS, and Sen1Floods11 configs freeze the OLMoEarth
backbone for linear-probe reproduction and train only the segmentation head.

## AWF and Nandi

AWF and Nandi are rslearn project datasets. Convert their rslearn dataset
directories once, then train from the generated manifests:

```bash
python projects/olmoearth/tools/convert_rslearn_seg.py \
  --dataset awf \
  --input-root /path/to/olmoearth_projects_awf_dataset \
  --output-root data/olmoearth_mmseg/awf

python projects/olmoearth/tools/convert_rslearn_seg.py \
  --dataset nandi \
  --input-root /path/to/olmoearth_projects_nandi_dataset \
  --output-root data/olmoearth_mmseg/nandi
```

The converter materializes raw Sentinel-2, label, valid-mask, and timestamp
arrays. The MMSeg configs then apply the same rslearn-style `Pad(size=31,
mode="center")`, `Crop(crop_size=16)`, flip, and OLMoEarth normalization steps
in the OpenMMLab pipeline.

Before training, smoke-check a converted split:

```bash
python projects/olmoearth/tools/check_converted_dataset.py \
  --data-root data/olmoearth_mmseg/awf \
  --ann-file train.json
```

This checker reads `metainfo.json` when present and validates sampled label
values against `num_classes` and `ignore_index`.

Then check that the OpenMMLab pipeline itself produces aligned tensors:

```bash
python projects/olmoearth/tools/check_pipeline.py \
  projects/olmoearth/configs/awf/olmoearth-base_4xb4-100e_awf-s2.py \
  --split train \
  --cfg-options \
  train_dataloader.dataset.data_root=data/olmoearth_mmseg/awf
```

Finally, verify one model loss step with the configured checkpoint:

```bash
python projects/olmoearth/tools/check_forward.py \
  projects/olmoearth/configs/awf/olmoearth-base_4xb4-100e_awf-s2.py \
  --split train \
  --device cuda \
  --cfg-options \
  model.backbone.model_config_path=/path/to/olmoearth/config.json \
  model.backbone.init_cfg.checkpoint=/path/to/olmoearth/weights.pth
```

## RGB Compatibility

RGB is supported only as an explicit adapter through `RGBToOlmoEarthS2`. It maps
R/G/B to Sentinel-2 B04/B03/B02 and fills missing Sentinel-2 bands with
normalized zero. This is not a paper-reproduction path.

## GEO-Bench S2 Segmentation

The GEO-Bench Sentinel-2 segmentation configs cover the segmentation tasks used
by OLMoEarth pretrain's GeoBench evaluator: `m-SA-crop-type` and
`m-cashew-plant`. The loader reads the official 13 GEO-Bench Sentinel-2 bands,
applies the task's imputation rules, applies the OLMoEarth
`NORM_NO_CLIP_2_STD` normalization from task band statistics, then selects the
12-band OLMoEarth Sentinel-2 L2A order.

Set `geobench_root` and `olmoearth_model_dir` at the top of each config before
running. The recommended aligned routes are offline embedding linear probing
and full finetuning.

For the faster paper-style offline linear probe, first extract dense
OLMoEarth embeddings once:

```bash
python projects/olmoearth/tools/extract_embeddings.py
```

Common extraction arguments can be edited directly in `SCRIPT_DEFAULTS` at the
top of `projects/olmoearth/tools/extract_embeddings.py`. CLI arguments still
override those defaults when you need a one-off change.

This extraction step is intentionally not a normal MMSeg training loop. It
reuses MMSeg config parsing, registries, datasets, and pipelines to get the
same samples that online training would see, but it bypasses the MMSeg Runner
and calls the OLMoEarth backbone in inference mode to materialize
`embedding.tif`. The following offline probe training is normal MMSeg again:
`OlmoEarthFeatureBackbone` reads the frozen embeddings and the decode head is
trained by `tools/train.py`.

Regenerate embeddings whenever the OLMoEarth checkpoint, `patch_size`, input
pipeline, split, crop size, or normalization changes. Otherwise the offline
probe may silently train on stale features.

The extractor also supports single-node multi-GPU sharding with `torchrun`.
Each rank writes its own temporary rank manifest, and rank 0 merges them into
the final `train.json`, `val.json`, `test.json`, and `summary.json`:

```bash
torchrun --nproc_per_node=4 projects/olmoearth/tools/extract_embeddings.py
```

`--tile-size` is optional. When it is greater than zero, samples larger than
that size are extracted with sliding-window tiles and merged back into one
`embedding.tif` with weighted overlap blending. Tile size and tile stride
(`tile_size - tile_overlap`) must be divisible by the OLMoEarth feature stride,
usually the config's `patch_size`. Use `--tile-overlap 0` for the fastest
extraction, or set a small overlap such as 64 pixels when boundary smoothness is
more important than speed.

`--save-raw-inputs` is optional. It writes `raw_input.tif` next to each
`embedding.tif` and `label.tif` for inspection. For crop-type this is the
13-band GEO-Bench Sentinel-2 image before OLMoEarth normalization. If you also
want the exact tensor fed to OLMoEarth after the MMSeg extraction pipeline, add
`--save-inputs` to write `input.tif`.

Then train only the patch-linear probe from the extracted embedding GeoTIFFs:

```bash
python tools/train.py \
  projects/olmoearth/configs/crop_type/olmoearth-base_1xb8-50e_crop-type-s2-offline-linear.py

python tools/train.py \
  projects/olmoearth/configs/cashew_plant/olmoearth-base_1xb8-50e_m-cashew-plant-s2-offline-linear.py
```

This is much closer to the original OLMoEarth evaluator: encoder forward is
paid once during extraction, and the 50-epoch probe training loop no longer
recomputes the OLMoEarth backbone.

The `*-s2-linear.py` configs are kept as convenient extraction-source configs
for `extract_embeddings.py`; they are not the recommended training route.

For full finetuning, use the configs aligned to `olmoearth_pretrain`'s finetune
evaluator. They keep the backbone frozen for the first 20% of training, then
unfreeze it:

```bash
python tools/train.py \
  projects/olmoearth/configs/crop_type/olmoearth-base_1xb8-50e_crop-type-s2-ft.py

python tools/train.py \
  projects/olmoearth/configs/cashew_plant/olmoearth-base_1xb4-50e_m-cashew-plant-s2-ft.py
```

## Potsdam

Potsdam support uses the RGB compatibility path, so it should be treated as an
out-of-domain OpenMMLab experiment rather than an OLMoEarth paper reproduction.
First prepare the official ISPRS data with MMSegmentation's converter:

```bash
python tools/dataset_converters/potsdam.py \
  /path/to/potsdam_zips \
  --out_dir data/potsdam
```

The Potsdam config uses `OlmoEarthPotsdamDataset`, which directly reads
`data/potsdam/img_dir/{train,val}/*.png` and
`data/potsdam/ann_dir/{train,val}/*.png`. It follows MMSeg's official Potsdam
label convention: label value `0` is ignored black boundary, and class ids
`1..6` are remapped to `0..5` through `reduce_zero_label=True`.

Train with:

```bash
python tools/train.py \
  projects/olmoearth/configs/potsdam/olmoearth-base_4xb4-50e_potsdam-rgb.py \
  --cfg-options \
  model.backbone.model_config_path=/path/to/olmoearth/config.json \
  model.backbone.init_cfg.checkpoint=/path/to/olmoearth/weights.pth
```

Two UPerNet-style Potsdam configs are also available. They follow OpenMMLab's
Potsdam/UPerNet convention more closely with 512 crops, `MultiLevelNeck`,
`UPerHead`, an auxiliary `FCNHead`, 80k iterations, `InfiniteSampler`, and
`PolyLR`. The `p4` version keeps the OLMoEarth feature at 1/4 resolution before
the neck; the `p16` version tests a coarser 1/16 feature for lower encoder
memory use.

```bash
python tools/train.py \
  projects/olmoearth/configs/potsdam/olmoearth-base_upernet_4xb4-80k_potsdam-rgb-p4-512x512.py \
  --cfg-options \
  model.backbone.model_config_path=/path/to/olmoearth/config.json \
  model.backbone.init_cfg.checkpoint=/path/to/olmoearth/weights.pth

python tools/train.py \
  projects/olmoearth/configs/potsdam/olmoearth-base_upernet_4xb4-80k_potsdam-rgb-p16-512x512.py \
  --cfg-options \
  model.backbone.model_config_path=/path/to/olmoearth/config.json \
  model.backbone.init_cfg.checkpoint=/path/to/olmoearth/weights.pth
```
