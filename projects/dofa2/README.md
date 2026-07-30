# DOFAv2 for MMSegmentation

This project integrates the DOFAv2 ViT backbone with UPerNet and provides
recipes for GEO-Bench m-cashew-plant, SVDT, and Ningbo-2m.

## What was corrected

- Official DOFA checkpoints are loaded once into the timm ViT. A missing
  `init_cfg` no longer calls the checkpoint loader with `None`.
- `freeze_backbone` explicitly controls full-backbone freezing. Frozen
  backbones remain in evaluation mode, so DropPath is disabled as expected.
- Patch-14 is preserved. `convert_patch_14_to_16=False` is the segmentation
  default used by the DOFAv2 reference implementation; the conversion remains
  available only as an opt-in compatibility option.
- m-cashew-plant uses the official nine Sentinel-2 bands, in this order:
  R, G, B, three red-edge bands, NIR, SWIR1, SWIR2.
- GEO-Bench objects are cached inside each transform/worker rather than in a
  split-overwriting module global.
- RGB/BGR handling is explicit. GEO-Bench and rasterio inputs stay in file
  band order; SVDT's MMSeg image loader is converted from BGR to RGB.
- Training uses random crop/rotation/flips, three warm-up epochs, gradient
  clipping, and conservative full-finetuning learning rates.
- The configurations use the server's existing absolute dataset and
  pretrained-weight paths. No path environment variables are required.

## Dependencies

Install MMSegmentation first, then install the project extras:

```bash
pip install -r projects/dofa2/requirements.txt
```

The pretrained checkpoint is:

```text
/mnt/ht2-nas2/EO_test/openmmlab-archive/pretrained/dofav2_vit_large_e150.pth
```

The configured dataset directories are:

```text
/mnt/ht2-nas2/EO_test/openmmlab-archive/dat/geo-bench-1.0/segmentation_v1.0/m-cashew-plant
/mnt/ht2-nas2/EO_test/openmmlab-archive/dat/SVDT
/mnt/ht2-nas2/EO_test/openmmlab-archive/dat/ningbo-slices-512-dataset-7class
```

## Configurations

| Dataset | Input | Frozen backbone | Full finetuning |
| --- | --- | --- | --- |
| m-cashew-plant | Sentinel-2, 9 bands, 224x224 | `dofav2-large_1xb12-20e_m-cashew-s2-frozen.py` | `dofav2-large_1xb12-20e_m-cashew-s2-finetune.py` |
| SVDT | RGB, 512x512 | `dofav2-large_1xb4-50e_svdt-rgb-frozen.py` | `dofav2-large_1xb4-50e_svdt-rgb-finetune.py` |
| Ningbo-2m | RGB, 512x512 | `dofav2-large_1xb4-50e_ningbo-rgb-frozen.py` | `dofav2-large_1xb4-50e_ningbo-rgb-finetune.py` |

The `1xbN` part is the per-GPU batch size. For multi-GPU runs, either keep the
learning rate fixed or enable `auto_scale_lr` after choosing a reference global
batch size.

## Train and test

Use the training and testing scripts already provided by MMSegmentation.
Single-GPU training:

```bash
python tools/train.py \
  projects/dofa2/configs/dofav2-large_1xb12-20e_m-cashew-s2-frozen.py
```

Eight GPUs:

```bash
bash tools/dist_train.sh \
  projects/dofa2/configs/dofav2-large_1xb12-20e_m-cashew-s2-frozen.py \
  8
```

Evaluation:

```bash
python tools/test.py CONFIG.py CHECKPOINT.pth
```

Use `--cfg-options` after the positional arguments for temporary overrides.
