# Shared Remote-Sensing Large Image Inference

This project contains the shared sliding-window GeoTIFF inference core used by
remote-sensing MMSegmentation projects.

The main entry point is:

```bash
python projects/rs_large_infer/tools/large_image_inference.py \
  /path/to/large.tif \
  /path/to/config.py \
  /path/to/checkpoint.pth \
  /path/to/pred_label.tif \
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
