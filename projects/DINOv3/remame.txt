[Test] project_root: /mnt/ht2-nas2/wj/mm_series/mmseg_olmoearth/projects/DINOv3
[Test] mmseg_root: /mnt/ht2-nas2/wj/mm_series/mmseg_olmoearth
[Test] dinov3_repo: /mnt/ht2-nas2/wj/mm_series/mmseg_olmoearth/projects/DINOv3
[Test] weights_path: /mnt/ht2-nas2/EO_test/dataset/dinov3_pretrained/sat493m/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
[Test] Build config:
  type: DINOv3ViT
  model_name: dinov3_vit16
  repo_path: /mnt/ht2-nas2/wj/mm_series/mmseg_olmoearth/projects/DINOv3
  weights_path: /mnt/ht2-nas2/EO_test/dataset/dinov3_pretrained/sat493m/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
  weights_name: SAT493M
  out_indices: (5, 11, 17, 23)
  patch_size: 16
  load_strict: True
  freeze: False
Traceback (most recent call last):
  File "/mnt/ht2-nas2/wj/mm_series/mmseg_olmoearth/projects/DINOv3/test/test_load_dinov3_backbone.py", line 274, in <module>
    main()
  File "/mnt/ht2-nas2/wj/mm_series/mmseg_olmoearth/projects/DINOv3/test/test_load_dinov3_backbone.py", line 258, in main
    model = build_model(args, project_root, dinov3_repo)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/ht2-nas2/wj/mm_series/mmseg_olmoearth/projects/DINOv3/test/test_load_dinov3_backbone.py", line 93, in build_model
    model = MODELS.build(cfg)
            ^^^^^^^^^^^^^^^^^
  File "/mnt/ht2-nas2/wj/mm_series/mmengine/mmengine/registry/registry.py", line 570, in build
    return self.build_func(cfg, *args, **kwargs, registry=self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/ht2-nas2/wj/mm_series/mmengine/mmengine/registry/build_functions.py", line 232, in build_model_from_cfg
    return build_from_cfg(cfg, registry, default_args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/ht2-nas2/wj/mm_series/mmengine/mmengine/registry/build_functions.py", line 121, in build_from_cfg
    obj = obj_cls(**args)  # type: ignore
          ^^^^^^^^^^^^^^^
  File "/mnt/ht2-nas2/wj/mm_series/mmseg_olmoearth/projects/DINOv3/models/dinov3_vit.py", line 89, in __init__
    self.dinov3 = self._build_dinov3_model()
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/ht2-nas2/wj/mm_series/mmseg_olmoearth/projects/DINOv3/models/dinov3_vit.py", line 163, in _build_dinov3_model
    raise ValueError(
ValueError: Unsupported DINOv3 model_name: dinov3_vit16. Available candidates: ['dinov3_convnext_base', 'dinov3_convnext_large', 'dinov3_convnext_small', 'dinov3_convnext_tiny', 'dinov3_vit7b16', 'dinov3_vitb16', 'dinov3_vith16plus', 'dinov3_vitl16', 'dinov3_vitl16plus', 'dinov3_vits16', 'dinov3_vits16plus']
