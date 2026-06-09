import argparse
import sys
from pathlib import Path

import torch


def parse_out_indices(text):
    """Parse out_indices from a string, e.g. '5,11,17,23'."""
    if text is None or text == '':
        return (5, 11, 17, 23)
    return tuple(int(item.strip()) for item in text.split(',') if item.strip())


def find_default_dinov3_repo(project_root):
    """Find the official DINOv3 source root.

    Recommended layout:

        mmsegmentation/
        └── projects/
            └── DINOv3/
                ├── models/
                ├── test/
                └── dinov3/          # official DINOv3 repo after unzip
                    └── dinov3/
                        └── hub/
                            └── backbones.py

    The returned path should be the official DINOv3 source root, i.e. the
    directory whose child contains `dinov3/hub/backbones.py`.
    """
    candidates = [
        project_root / 'dinov3',
        project_root,
        project_root / 'third_party' / 'dinov3-main',
        project_root.parent.parent / 'third_party' / 'dinov3-main',
    ]

    for candidate in candidates:
        if (candidate / 'dinov3' / 'hub' / 'backbones.py').is_file():
            return candidate.resolve()

    return None


def add_python_paths(project_root, mmseg_root, dinov3_repo):
    """Add MMSegmentation root and official DINOv3 source root to sys.path."""
    paths = [mmseg_root, dinov3_repo]

    for path in paths:
        if path is None:
            continue
        path = str(Path(path).resolve())
        if path not in sys.path:
            sys.path.insert(0, path)


def import_project_modules():
    """Import project modules so that DINOv3ViT is registered into MMSeg."""
    try:
        import projects.DINOv3.models  # noqa: F401
    except Exception as exc:
        raise ImportError(
            'Failed to import projects.DINOv3.models. Please run this script '
            'inside the MMSegmentation repository, or pass --mmseg-root to the '
            'MMSegmentation root directory.'
        ) from exc


def build_model(args, project_root, dinov3_repo):
    from mmseg.registry import MODELS

    cfg = dict(
        type=args.backbone_type,
        model_name=args.model_name,
        repo_path=str(dinov3_repo),
        weights_path=str(Path(args.weights_path).expanduser().resolve()),
        weights_name=args.weights_name,
        out_indices=parse_out_indices(args.out_indices),
        patch_size=args.patch_size,
        load_strict=not args.non_strict,
        freeze=args.freeze,
    )

    print('[Test] Build config:')
    for key, value in cfg.items():
        print(f'  {key}: {value}')

    model = MODELS.build(cfg)
    return model


def run_forward_test(model, args):
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    model = model.to(device)
    model.eval()

    h, w = args.input_size
    dummy = torch.randn(args.batch_size, 3, h, w, device=device)

    with torch.no_grad():
        outputs = model(dummy)

    print('[Test] Forward success.')
    print(f'[Test] Number of output feature maps: {len(outputs)}')
    for idx, feat in enumerate(outputs):
        print(f'  output[{idx}]: shape={tuple(feat.shape)}, dtype={feat.dtype}')


def main():
    parser = argparse.ArgumentParser(
        description='Test DINOv3 backbone registration and local checkpoint loading in MMSegmentation.'
    )

    parser.add_argument(
        '--weights-path',
        required=True,
        help='Path to the local DINOv3 checkpoint, e.g. dinov3_vitl16_pretrain_sat493m.pth.'
    )
    parser.add_argument(
        '--model-name',
        default='dinov3_vitl16',
        help=(
            'Official DINOv3 backbone builder name. Examples: '
            'dinov3_vits16, dinov3_vitb16, dinov3_vitl16, '
            'dinov3_vitl16plus, dinov3_vith16plus.'
        )
    )
    parser.add_argument(
        '--weights-name',
        default='SAT493M',
        choices=['LVD1689M', 'SAT493M'],
        help=(
            'Official DINOv3 weight type used when building the architecture. '
            'Use SAT493M for large-sat weights; use LVD1689M for normal DINOv3 weights.'
        )
    )
    parser.add_argument(
        '--backbone-type',
        default='DINOv3ViT',
        help='Registered MMSeg backbone type. Default: DINOv3ViT.'
    )
    parser.add_argument(
        '--dinov3-repo',
        default=None,
        help=(
            'Path to the official DINOv3 source root. If omitted, the script '
            'will try to find it under projects/DINOv3/dinov3.'
        )
    )
    parser.add_argument(
        '--mmseg-root',
        default=None,
        help='Path to MMSegmentation root. If omitted, infer it from projects/DINOv3/test.'
    )
    parser.add_argument(
        '--project-root',
        default=None,
        help='Path to projects/DINOv3. If omitted, infer it from this test file.'
    )
    parser.add_argument(
        '--out-indices',
        default='5,11,17,23',
        help='Transformer block indices to export, e.g. "2,5,8,11" or "5,11,17,23".'
    )
    parser.add_argument(
        '--patch-size',
        type=int,
        default=16,
        help='Patch size of DINOv3 ViT. Default: 16.'
    )
    parser.add_argument(
        '--non-strict',
        action='store_true',
        help='Load checkpoint with strict=False.'
    )
    parser.add_argument(
        '--freeze',
        action='store_true',
        help='Freeze DINOv3 backbone after loading.'
    )
    parser.add_argument(
        '--run-forward',
        action='store_true',
        help='Run a dummy forward test after loading. Default only tests checkpoint loading.'
    )
    parser.add_argument(
        '--input-size',
        type=int,
        nargs=2,
        default=[224, 224],
        metavar=('HEIGHT', 'WIDTH'),
        help='Dummy input size for --run-forward. Both values should be divisible by patch size.'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Dummy batch size for --run-forward.'
    )
    parser.add_argument(
        '--device',
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device for --run-forward. Default: auto.'
    )

    args = parser.parse_args()

    this_file = Path(__file__).resolve()

    if args.project_root is not None:
        project_root = Path(args.project_root).expanduser().resolve()
    else:
        # Expected path: mmsegmentation/projects/DINOv3/test/test_load_dinov3_backbone.py
        project_root = this_file.parents[1]

    if args.mmseg_root is not None:
        mmseg_root = Path(args.mmseg_root).expanduser().resolve()
    else:
        # Expected path: mmsegmentation/projects/DINOv3
        if project_root.parent.name == 'projects':
            mmseg_root = project_root.parent.parent.resolve()
        else:
            mmseg_root = Path.cwd().resolve()

    if args.dinov3_repo is not None:
        dinov3_repo = Path(args.dinov3_repo).expanduser().resolve()
    else:
        dinov3_repo = find_default_dinov3_repo(project_root)

    if dinov3_repo is None:
        raise FileNotFoundError(
            'Cannot find official DINOv3 source root. Please pass --dinov3-repo. '
            'The expected source root should contain dinov3/hub/backbones.py.'
        )

    weights_path = Path(args.weights_path).expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {weights_path}')

    print(f'[Test] project_root: {project_root}')
    print(f'[Test] mmseg_root: {mmseg_root}')
    print(f'[Test] dinov3_repo: {dinov3_repo}')
    print(f'[Test] weights_path: {weights_path}')

    add_python_paths(project_root, mmseg_root, dinov3_repo)
    import_project_modules()

    model = build_model(args, project_root, dinov3_repo)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('[Test] Checkpoint load success.')
    print(f'[Test] Total parameters: {total_params:,}')
    print(f'[Test] Trainable parameters: {trainable_params:,}')

    if args.run_forward:
        run_forward_test(model, args)
    else:
        print('[Test] Forward test skipped. Add --run-forward if you want to verify output feature maps.')


if __name__ == '__main__':
    main()
