MADOS uses dataset-specific normalization in the OLMoEarth eval code:
`norm_stats_from_pretrained=False` with `norm_no_clip_2_std`. The runnable
config in this directory uses `OlmoEarthDatasetNormalize` instead of pretraining
computed statistics.

`OlmoEarthIoUMetric` reports MMSeg-style `aAcc`, `mIoU`, and `mAcc` by
default. Add `iou_metrics=["mIoU", "mFscore"]` in the evaluator to also log
`mFscore`, `mPrecision`, and `mRecall`.
