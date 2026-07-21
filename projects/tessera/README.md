# TESSERA for MMSegmentation

Independent MMSegmentation project for TESSERA dense embeddings and the
fixed-sampling v1-style online backbone.

- `configs/tessera/*offline-linear.py`: train a segmentation probe on
  precomputed 128-D TESSERA embeddings, including v1.1 QAT `int8 + scale`.
- `configs/tessera/*online-linear.py`: encode standard TESSERA temporal arrays
  during training with fixed 40-observation S1/S2 sampling.

Both configs use `projects.tessera.tessera` and have no OLMoEarth dependency.
