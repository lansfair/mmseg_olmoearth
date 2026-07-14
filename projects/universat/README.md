# UniverSat for MMSegmentation 1.x

This project migrates the UniverSat multimodal Earth-observation encoder into
[MMSegmentation 1.x](https://github.com/open-mmlab/mmsegmentation) as an
external project under `mmsegmentation/projects/universat/`.

## Layout

```
projects/universat/
├── universat/                         # Python package
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbones/
│   │   │   ├── __init__.py
│   │   │   ├── universat_backbone.py  # MMSeg 1.x backbone wrapper
│   │   │   └── universat_modules/     # Original encoder code (copied)
│   │   │       ├── UniverSat.py
│   │   │       ├── UniversalPatchEncoder.py
│   │   │       ├── modality_registry.py
│   │   │       ├── masking/
│   │   │       └── utils/
│   │   ├── decode_heads/
│   │   │   ├── universat_seg_head.py
│   │   │   └── universat_lp_head.py
│   │   └── data_preprocessors.py
│   └── datasets/
│       ├── __init__.py
│       ├── universat_dataset.py
│       └── transforms.py
├── configs/
│   └── base_universat_seg.py          # Generic JSON-dataset template
├── pastis/                            # Standalone PASTIS-R project
│   ├── universat_pastis/
│   ├── configs/
│   │   ├── universat-base_pastis_lp.py  # PASTIS-R linear probe
│   │   └── universat-base_pastis_ft.py  # PASTIS-R fine-tune
│   ├── train.sh
│   └── test.sh
├── train.sh
└── test.sh
```

## Requirements

- MMSegmentation 1.x / OpenMMLab 2.0 (uses `mmengine` + `mmseg.registry`)
- PyTorch >= 2.0 (eager mode is the stable default; set
  `compile_encoder=True` to opt into dynamic compilation)
- `safetensors` (for loading released `.safetensors` checkpoints)
- `einops` (used by `flexiVit.py`)

```bash
pip install -r projects/universat/requirements.txt
```

## PASTIS-R downstream evaluation

For a complete PASTIS-R linear-probe / fine-tuning project that follows the
same layout as `projects/copernicus/pastis`, see the `pastis/` subdirectory.
It contains a dedicated dataset class (`UniverSatPASTISDataset`), a custom
collate function for variable-length time series, and ready-to-use configs.

## Usage

### 1. Prepare data

Create a JSON split file for your dataset::

```json
[
  {
    "filenames": {
      "s2": "s2/xxx.npy",
      "s1": "s1/xxx.npy"
    },
    "dates": {
      "s2": [365, 377, 389],
      "s1": [366, 378, 390]
    },
    "ann": {"seg_map": "masks/xxx.png"},
    "height": 360,
    "width": 360
  }
]
```

Copy `configs/base_universat_seg.py` and update its `data_root`, split paths,
`num_classes`, `ignore_index`, and per-modality `mean`/`std` statistics.

### 2. Prepare checkpoint

Put the pretrained UniverSat checkpoint (`.safetensors` or `.pth`) in the path
referenced by `MM_ARCHIVE_CKPT_HOME`, and make sure the config's
`init_cfg.checkpoint` points to it.

### 3. Train

From the MMSegmentation root (the folder containing `tools/`):

```bash
export MM_ARCHIVE_DATA_HOME=/path/to/data
export MM_ARCHIVE_CKPT_HOME=/path/to/checkpoints
bash projects/universat/train.sh \
    projects/universat/pastis/configs/universat-base_pastis_lp.py
```

Or use the provided launcher::

```bash
cd projects/universat
bash train.sh
```

### 4. Test

```bash
bash projects/universat/test.sh \
    projects/universat/pastis/configs/universat-base_pastis_lp.py \
    path/to/checkpoint.pth
```

## Key components

- `UniverSatBackbone`: registered as `MODELS`, wraps the original encoder and
  exposes MMSeg-style multi-scale features.
- `UniverSatSegHead`: small conv-based segmentation head.
- `UniverSatLinearProbeHead`: LayerNorm + 1x1 classifier for linear probing.
- `UniverSatDataPreprocessor`: passes the multimodal dict through to the
  backbone.
- `UniverSatSegDataset` + `LoadMultimodalFromFile`/`NormalizeMultimodal`/
  `PackUniverSatInputs`: multimodal data loading for MMSeg 1.x.

## Adapting to your own dataset

1. Add your modality names to `modalities` in both the dataset config and the
   backbone config.
2. If a modality is not in `modality_registry.py`, provide `wavelengths`,
   `input_res`, and `subpatches` overrides in the backbone config.
3. Replace `mean`/`std` in the dataset config with values computed from your
   training split.
4. Adjust `output_grid`, `patch_size`, and `crop_size` so they are consistent
   with your input patch layout.
   `output_grid` is the side length: `36` means a 36 x 36 token grid.
