# DOFAv2 for MMSegmentation

This project integrates the DOFAv2 ViT backbone with UPerNet and provides
recipes for GEO-Bench m-cashew-plant, SVDT, and Ningbo-2m.

## What was corrected

- Official DOFA checkpoints are loaded once into the timm ViT. A missing
  `init_cfg` no longer calls the checkpoint loader with `None`.
- `freeze_backbone` explicitly controls full-backbone freezing. Frozen
  backbones remain in evaluation mode, so DropPath is disabled as expected.
- The retained m-cashew-plant recipe uses the best verified setup: RGB input,
  256x256 crops, and `convert_patch_14_to_16=True`.
- GEO-Bench objects are cached inside each transform/worker rather than in a
  split-overwriting module global.
- RGB/BGR handling is explicit. GEO-Bench and rasterio inputs stay in file
  band order; SVDT's MMSeg image loader is converted from BGR to RGB.
- The m-cashew recipe freezes the backbone and trains UPerNet for 30 epochs
  with center crop, rotation, flips, AdamW at 5e-3, and cosine decay.
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
| m-cashew-plant | RGB, 256x256, P16 | `dofav2-large_4xb16-30e_m-cashew-rgb-p16-256.py` | - |
| SVDT | RGB, 512x512 | `dofav2-large_1xb4-50e_svdt-rgb-frozen.py` | `dofav2-large_1xb4-50e_svdt-rgb-finetune.py` |
| Ningbo-2m | RGB, 512x512 | `dofav2-large_1xb4-50e_ningbo-rgb-frozen.py` | `dofav2-large_1xb4-50e_ningbo-rgb-finetune.py` |

The retained m-cashew configuration uses four GPUs with 16 samples per GPU,
for the verified global batch size of 64. It reached 64.53 validation mIoU and
58.96 test mIoU in the comparison run.

The `1xbN` part is the per-GPU batch size. For multi-GPU runs, either keep the
learning rate fixed or enable `auto_scale_lr` after choosing a reference global
batch size.

## Train and test

Use the training and testing scripts already provided by MMSegmentation.
Four-GPU m-cashew training:

```bash
bash tools/dist_train.sh \
  projects/dofa2/configs/dofav2-large_4xb16-30e_m-cashew-rgb-p16-256.py \
  4
```

Evaluation:

```bash
python tools/test.py CONFIG.py CHECKPOINT.pth
```

Use `--cfg-options` after the positional arguments for temporary overrides.
