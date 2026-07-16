# Unified GeoFM Embedding Adapters

This project provides one input and output contract for Earth-observation
foundation models used in embedding extraction and MMSegmentation probes.

## Contracts

Canonical multimodal input:

```python
inputs = {
    "modalities": {
        "sentinel1": s1,          # [B, T, C, H, W]
        "sentinel2_l2a": s2,     # [B, T, C, H, W]
    },
    "timestamps": timestamps,    # [B, T, 3], day/month/year
    "masks": {},                 # optional native model masks
}
```

Canonical outputs:

- global embedding: `[B, D]`
- dense embedding: `[B, D, Hf, Wf]`

Adapters declare supported and required modalities, output modes, temporal
support, multimodal support, and native feature stride. Invalid combinations
fail before model forward.

## Unified linear probes

Dense segmentation probes use the same three-part contract for every model:

1. `GeoFMBackbone` converts the adapter output to `[B,D,Hf,Wf]` and is the
   single owner of the frozen state.
2. `GeoFMLinearHead` applies one trainable 1x1 affine classifier. Bilinear
   resizing contains no trainable parameters.
3. `GeoFMFreezeBackboneHook` audits trainability and keeps a frozen backbone
   in evaluation mode, including its normalization and dropout layers.

Use two separately reported protocols:

| Track | Head | Purpose |
| --- | --- | --- |
| fair | `GeoFMLinearHead` | Compare every model with the same probe form |
| reference | model-paper head | Reproduce a model's published protocol |

For OlmoEarth, the reference head is `GeoFMPatchLinearHead`. It exactly maps
each patch embedding to `classes * patch_size^2` logits and rearranges those
logits to pixels. Do not use it as the default fair head for models with
different native strides, because its parameter count grows with
`patch_size^2`.

Strict linear probing is configured as follows:

```python
model = dict(
    type="GeoFMEncoderDecoder",
    backbone=dict(
        type="GeoFMBackbone",
        adapter=adapter,
        output_mode="dense",
        frozen=True,
    ),
    decode_head=dict(
        type="GeoFMLinearHead",
        in_channels=embedding_dim,
        channels=embedding_dim,
        in_index=0,
        num_classes=num_classes,
        scale_factor=feature_stride,
        dropout_ratio=0,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False),
    ),
)
custom_hooks = [
    dict(
        type="GeoFMFreezeBackboneHook",
        unfreeze_epoch=None,
        strict=True,
    )
]
```

Set `unfreeze_epoch=0` for full fine-tuning or a positive epoch for staged
fine-tuning. Those runs must be reported as fine-tuning, not linear probing.
The optimizer, schedule, data split, augmentation, loss, random seeds, and
feature-to-label alignment must remain identical across compared models.

For precomputed dense embeddings, replace only the backbone:

```python
backbone=dict(
    type="PrecomputedEmbeddingBackbone",
    out_channels=embedding_dim,
)
```

The same decode head, loss, hook, dataloaders, and metrics are then reused.
This makes released embedding products comparable with models that can run
online without pretending that a product-only model can be fine-tuned.

Timestamp month indexing is explicit per adapter. OlmoEarth-native inputs use
zero-based months. `TESSERAAdapter(timestamp_month_base=0)` converts those
timestamps to one-based calendar months before calculating day of year.

## OlmoEarth

`OlmoEarthAdapter` supports the released OlmoEarth variants through the same
encoder output and `pool_unmasked_tokens` implementation used by the official
evaluation code. It supports joint multimodal forward and both global and
dense embeddings. Dense outputs are converted from official `[B,H,W,D]` to
MMSegmentation `[B,D,H,W]`.

Example backbone fragment:

```python
custom_imports = dict(
    imports=[
        "projects.olmoearth.olmoearth",
        "projects.geofm_embeddings.geofm_embeddings",
    ],
    allow_failed_imports=False,
)

model = dict(
    type="GeoFMEncoderDecoder",
    backbone=dict(
        type="GeoFMBackbone",
        output_mode="dense",
        adapter=dict(
            type="OlmoEarthAdapter",
            model_config_path=(
                "checkpoints/geofm/olmoearth/base/config.json"
            ),
            init_cfg=dict(
                type="Pretrained",
                checkpoint=(
                    "checkpoints/geofm/olmoearth/base/weights.pth"
                ),
            ),
            model_variant="base",
            modalities=["sentinel2_l2a"],
            num_timesteps=12,
            patch_size=4,
            pooling_type="mean",
            out_channels=768,
        ),
    ),
)
```

For one configured modality, the adapter also accepts the existing OlmoEarth
MMSeg tensor layout `[B,C*T,H,W]`. New multimodal pipelines should use the
canonical dictionary input.

## Extraction

The generic extractor builds the full MMSeg config so it reuses the configured
dataset pipeline and data preprocessor:

```bash
python projects/geofm_embeddings/tools/extract_embeddings.py \
  path/to/config.py work_dirs/geofm_embeddings/olmoearth_pastis \
  --split test --mode dense --dense-format geotiff
```

Use `--mode global` for `[B,D]` PT files. L2 normalization is disabled by
default to preserve native model output and can be enabled explicitly with
`--l2-normalize` for cosine-retrieval products.

Compare one exported embedding with an official reference:

```bash
python projects/geofm_embeddings/tools/compare_embeddings.py \
  official.pt exported.pt --output comparison.json
```

The comparison reports shape, MAE, RMSE, maximum absolute error, and mean/min
cosine similarity.

## Embedding quality evaluation

The project includes model-agnostic evaluation for already exported
embeddings. Install the evaluation-only dependencies without changing the
MMSeg runtime contract:

```bash
pip install -r projects/geofm_embeddings/requirements-evaluation.txt
```

The evaluator reads the `train.json`, `val.json`, and `test.json` manifests
written by `tools/extract_embeddings.py`. Global `[D]` PT tensors are evaluated
directly. Dense `[D,H,W]` PT or GeoTIFF tensors are converted to one patch
vector with `mean`, `mean_max`, or `stats` pooling. Spatially dense segmentation
evaluation remains in MMSeg through `GeoFMLinearHead` or the explicitly labeled
reference head.

Provide one metadata row per sample:

```text
sample_id,label,split,latitude,longitude
sample_0001,Forest,train,48.10,11.50
```

When each dense export directory also contains `label.tif`, scene labels can
be generated by majority valid-pixel voting:

```bash
python projects/geofm_embeddings/tools/build_evaluation_metadata.py \
  --root work_dirs/geofm_embeddings/olmoearth_pastis \
  --output work_dirs/geofm_embeddings/olmoearth_pastis/metadata.csv \
  --ignore-label -1
```

The output also records label purity. This majority-label metadata is suitable
for patch-level diagnostics; use the original masks and MMSeg metrics for dense
segmentation conclusions.

Copy and edit
`configs/evaluation/basic_representation.example.json`, then run:

```bash
python projects/geofm_embeddings/tools/evaluate_embeddings.py \
  projects/geofm_embeddings/configs/evaluation/my_experiment.json
```

The two recommended tracks are native dimensionality and train-fitted PCA-64.
The suite reports K-means/DBSCAN clustering, KNN, sklearn linear probing,
cosine semantic retrieval, intrinsic dimension, and PCA diagnostics. Retrieval
includes Hit@K, Precision@K, Recall@K, mAP@K, full-gallery MRR, and a per-query
result table. `sample_policy=strict` requires every compared model to contain
the same aligned sample IDs.

## Implementation Status

| Family | Adapter | Global | Dense | Status |
| --- | --- | --- | --- | --- |
| OlmoEarth | `OlmoEarthAdapter` | yes | yes | implemented |
| CopernicusFM | `CopernicusFMAdapter` | yes | yes | implemented |
| DINOv3 | `DINOv3Adapter` | yes | yes | implemented |
| TESSERA | `TESSERAAdapter` | yes | yes | implemented |
| CROMA | planned | yes | yes | pending |
| Galileo | planned | yes | yes | pending |
| PrithviV2 | planned | yes | yes | pending |
| TerraMind | planned | yes | yes | pending |
| Clay | planned | yes | yes | pending |
| Presto | planned | yes | yes | pending |
| Satlas | planned | yes | yes | pending |
| Panopticon | planned | yes | yes | pending |
| AnySat | planned | yes | yes | pending |

`projects/universat` is a separate UniverSat implementation and is not used as
an AnySat replacement.

The pending families can already be wired through
`OfficialOlmoEarthWrapperAdapter`, which directly instantiates their classes
under `olmoearth_pretrain.evals.models`. They remain marked pending until their
checkpoint-specific configs and numerical parity tests are added.

Example CROMA adapter:

```python
adapter = dict(
    type="OfficialOlmoEarthWrapperAdapter",
    preset="croma",
    model_variant="base",
    modalities=["sentinel1", "sentinel2_l2a"],
    out_channels=768,
    native_stride=8,
    wrapper_kwargs=dict(
        size="base",
        load_directory="checkpoints/croma",
    ),
)
```

Available presets are `anysat`, `clay`, `croma`, `galileo`, `panopticon`,
`presto`, `prithviv2`, `satlas`, and `terramind`. Constructor arguments are
passed unchanged to the official wrapper through `wrapper_kwargs`.
