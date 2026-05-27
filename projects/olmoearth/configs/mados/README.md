MADOS uses dataset-specific normalization in the OLMoEarth eval code:
`norm_stats_from_pretrained=False` with `norm_no_clip_2_std`. The runnable
config in this directory uses `OlmoEarthDatasetNormalize` instead of pretraining
computed statistics.

`OlmoEarthIoUMetric` reports both `mIoU` and `micro_f1` so the experiment can
track either paper/reporting convention without changing the evaluator.
