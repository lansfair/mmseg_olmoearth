# Shared Remote-Sensing Large Image Inference

This project contains the shared sliding-window GeoTIFF inference core used by
remote-sensing MMSegmentation projects.

The main entry point is:

Model/dataset-specific preset scripts can live in `projects/rs_large_infer/tools`
and pass defaults into the same shared runner. For example:

```bash
python projects/rs_large_infer/tools/infer_olmoearth_potsdam_rgb.py \
  --image /data/potsdam_large_rgb.tif \
  --checkpoint /checkpoints/olmoearth_potsdam.pth \
  --output /outputs/potsdam_pred.tif
```

```bash
python projects/rs_large_infer/tools/infer_dinov3_potsdam_rgb.py \
  --image /data/potsdam_large_rgb.tif \
  --checkpoint /checkpoints/dinov3_potsdam.pth \
  --output /outputs/potsdam_pred.tif
```

The generic entry point remains:

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

Project-specific copies are intentionally not kept. Use this single entry point
for standard models, OLMoEarth, and CopernicusFM.

Adapter selection:

```text
auto        Detect CopernicusFM, OLMoEarth RGB/S2, then fall back to standard.
standard    Read ordinary RGB/multiband GeoTIFF windows and run config transforms.
rgb         For OLMoEarth configs, use RGBToOlmoEarthS2; otherwise read bands 1-3.
s2          Use OLMoEarth Sentinel-2 band ordering and metadata.
copernicus  Generate Copernicus-FM [lon, lat, time, patch_area] metadata.
```

The shared core owns model loading, sliding-window grids, rasterio reads/writes,
batch inference, and GeoTIFF output. Adapters only decide which bands to read,
which transforms to keep, and which metainfo fields to attach.



