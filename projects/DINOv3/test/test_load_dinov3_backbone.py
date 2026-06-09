import argparse
import sys
from pathlib import Path

import torch


def find_mmseg_root() -> Path:
    """Find MMSegmentation root from this test file location.

    Expected location:

    mmsegmentation/projects/DINOv3/test/test_load_dinov3_backbone.py
    """

    cur = Path(__file__).resolve()
    for parent in cur.parents:
        if (parent / 'mmseg').is_dir() and (parent / 'tools').is_dir():
            return parent

    # Fallback for the standard location:
    # mmsegmentation/projects/DINOv3/test/xxx.py -> mmsegmentation
    return cur.parents[3]


def find_default_weights(project_root: Path):
    candidates = [
        project_root / 'checkpoints' / 'dinov3-vitl16-pretrain-sat493m.pth',
        project_root / 'checkpoints' / 'dinov3_vitl16_pretrain_sat493m.pth',
        project_root / 'dinov3-vitl16-pretrain-sat493m.pth',
        project_root / 'dinov3_vitl16_pretrain_sat493m.pth',
    ]

    for path in candidates:
        if path.is_file():
            return str(path)

    return None


def parse_out_indices(text: str):
    if not text:
        return None
    return tuple(int(item.strip()) for item in text.split(',') if item.strip())


def main():
    mmseg_root = find_mmseg_root()
    project_root = Path(__file__).resolve().parents[1]

    if str(mmseg_root) not in sys.path:
        sys.path.insert(0, str(mmseg_root))

    parser = argparse.ArgumentParser(
        description='Test loading DINOv3 backbone registered in MMSegmentation.')
    parser.add_argument(
        '--repo-path',
        default=str(project_root / 'dinov3'),
        help='Official DINOv3 repository root. Default: projects/DINOv3/dinov3')
    parser.add_argument(
        '--weights-path',
        default=find_default_weights(project_root),
        help='Local DINOv3 checkpoint path.')
    parser.add_argument(
        '--model-name',
        default='dinov3_vitl16',
        help='DINOv3 builder name, e.g. dinov3_vits16/dinov3_vitb16/dinov3_vitl16.')
    parser.add_argument(
        '--weights-name',
        default='SAT493M',
        help='DINOv3 weight enum name, e.g. LVD1689M or SAT493M.')
    parser.add_argument(
        '--out-indices',
        default='',
        help='Comma-separated block indices. Empty means auto.')
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Use strict checkpoint loading. Default: False for easier debugging.')
    args = parser.parse_args()

    if not args.weights_path:
        raise FileNotFoundError(
            'No checkpoint found. Please pass --weights-path, or put the file '
            'under projects/DINOv3/checkpoints/.'
        )

    # Importing this module triggers @MODELS.register_module().
    import projects.DINOv3.models  # noqa: F401
    from mmseg.registry import MODELS

    backbone_cfg = dict(
        type='DINOv3ViT',
        model_name=args.model_name,
        repo_path=args.repo_path,
        weights_path=args.weights_path,
        weights_name=args.weights_name,
        out_indices=parse_out_indices(args.out_indices),
        load_strict=args.strict,
        freeze=False,
    )

    model = MODELS.build(backbone_cfg)
    model.to(args.device)
    model.eval()

    x = torch.randn(args.batch_size, 3, args.height, args.width, device=args.device)

    with torch.no_grad():
        outs = model(x)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('DINOv3 backbone load test passed.')
    print(f'model_name: {args.model_name}')
    print(f'weights_name: {args.weights_name}')
    print(f'weights_path: {args.weights_path}')
    print(f'repo_path: {args.repo_path}')
    print(f'out_indices: {model.out_indices}')
    print(f'out_channels: {model.out_channels}')
    print(f'total_params: {total_params:,}')
    print(f'trainable_params: {trainable_params:,}')
    print(f'input_shape: {tuple(x.shape)}')

    for i, feat in enumerate(outs):
        print(f'out[{i}]: shape={tuple(feat.shape)}, dtype={feat.dtype}')


if __name__ == '__main__':
    main()
