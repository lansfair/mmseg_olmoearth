# CopernicusBench MMSegmentation Project

This project migrates Copernicus-Bench segmentation tasks into
MMSegmentation's `projects/` layout.

The dataset implementations read the original CSV splits and GeoTIFF files
directly. Copernicus-FM is registered as an MMSegmentation backbone,
`Feature2Pyramid` is used as the neck, and UPerHead/FCNHead are configured in
the standard `EncoderDecoder` style.

The project uses GDAL (`osgeo.gdal`) to read multi-band GeoTIFF files and
projection metadata. PASTIS uses the OlmoEarth-processed split layout.
PASTIS S2 tensors keep their 12 timesteps and 13 processed bands; the temporal
segmentor forwards each timestep with its own Copernicus-FM time metadata and
averages logits over time.

Supported configs:

```text
projects/CopernicusBench/configs/
  upernet_copernicus-fm-base_1xb16-50e_dfc2020-s2-256x256.py
  upernet_copernicus-fm-base_1xb16-50e_cloud-s2-512x512.py
  upernet_copernicus-fm-base_1xb16-50e_cloud-s3-224x224.py
  upernet_copernicus-fm-base_1xb16-50e_cloud-s3-binary-224x224.py
  upernet_copernicus-fm-base_1xb16-50e_lc100seg-s3-288x288.py
  linear-probe_copernicus-fm-base_1xb16-50e_pastis-processed-s2-64x64.py
  linear-finetune_copernicus-fm-base_1xb16-50e_pastis-processed-s2-64x64.py
  upernet_copernicus-fm-base_1xb16-50e_pastis-processed-s2-64x64.py
```

Expected data layouts:

```text
data/copernicusbench/dfc2020_s1s2/
  dfc/
  s2/
  dfc-train-new.csv
  dfc-val-new.csv
  dfc-test-new.csv

data/copernicusbench/cloud_s2/
  cloud/
  s2_toa/
  train.csv
  val.csv
  test.csv

data/copernicusbench/cloud_s3/
  cloud_binary/
  cloud_multi/
  s3_olci/
  train.csv
  val.csv
  test.csv

data/copernicusbench/lc100_s3/
  lc100/
  s3_olci/
  static_fnames-train.csv
  static_fnames-val.csv
  static_fnames-test.csv

data/pastis_r/
  pastis_r_train/
    s2_images/
    targets.pt
    months.pt
  pastis_r_valid/
    s2_images/
    targets.pt
    months.pt
  pastis_r_test/
    s2_images/
    targets.pt
    months.pt
```

Run from the MMSegmentation repository root:

```bash
python tools/train.py projects/CopernicusBench/configs/upernet_copernicus-fm-base_1xb16-50e_dfc2020-s2-256x256.py
python tools/train.py projects/CopernicusBench/configs/linear-probe_copernicus-fm-base_1xb16-50e_pastis-processed-s2-64x64.py
python tools/train.py projects/CopernicusBench/configs/linear-finetune_copernicus-fm-base_1xb16-50e_pastis-processed-s2-64x64.py
python tools/train.py projects/CopernicusBench/configs/upernet_copernicus-fm-base_1xb16-50e_pastis-processed-s2-64x64.py
```

## Large GeoTIFF Inference

Use `projects/rs_large_infer/src/cli.py` for
sliding-window inference on a large single GeoTIFF. The shared core handles
sliding windows, model inference, and GeoTIFF output; the Copernicus adapter
bypasses dataset-only file loaders, keeps test-time image transforms such as
`NormalizeMultibandImage`, and rebuilds Copernicus-FM metadata for every
window:

```text
copernicus_meta = [window_center_lon, window_center_lat, sensing_time, patch_area]
```

For Copernicus configs, `--input-mode auto` switches to Copernicus mode when it
sees `CopernicusEncoderDecoder`, `CopernicusFMBackbone`,
`LoadCopernicusGeoTiffImageFromFile`, or `AddCopernicusMeta`.

Example:

```bash
python projects/rs_large_infer/src/cli.py \
  /path/to/large_s2.tif \
  projects/CopernicusBench/configs/upernet_copernicus-fm-base_1xb16-50e_dfc2020-s2-256x256.py \
  /path/to/mmseg_checkpoint.pth \
  /path/to/pred_label.tif \
  --window-size 256 256 \
  --stride 128 128 \
  --batch-size 1 \
  --device cuda:0
```

The script reads `band_indices`, `band_scales`, `nan_to_num`, `to_float32`,
`date_separator`, `date_token_index`, and `patch_area` from
`LoadCopernicusGeoTiffImageFromFile` when that loader is present. For configs
that use `LoadSingleRSImageFromFile` plus `AddCopernicusMeta`, it reads all
bands and takes `patch_area` from `AddCopernicusMeta` or
`model.backbone.patch_area`.

Useful overrides:

```bash
--band-indices 1 2 3 4 5 6 7 8 9 10 11 12 13
--band-scales 0.0001 0.0001 0.0001
--copernicus-date 20240501
--copernicus-date-days 19844
--copernicus-lon-lat 116.3 39.9
--copernicus-patch-area 0.0256
```

`--copernicus-date` and `--copernicus-date-days` are useful when the large
image filename does not follow the dataset date convention. If the raster has a
CRS, lon/lat are computed from each window center and transformed to EPSG:4326.



