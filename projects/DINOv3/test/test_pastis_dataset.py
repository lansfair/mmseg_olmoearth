"""Smoke test for DINOv3/datasets PASTIS loading.

Run from the MMSegmentation root, for example:

    python projects/DINOv3/datasets/test_pastis_dataset.py \
        --data-root data/pastis_dataset_64 \
        --split pastis_r_train \
        --resize 128 128

You can also run a self-contained fake-data test:

    python projects/DINOv3/datasets/test_pastis_dataset.py --make-fake --resize 128 128
"""

from pathlib import Path
import argparse
import shutil
import sys
import tempfile

import torch


def _add_import_paths():
    this_file = Path(__file__).resolve()
    project_dir = this_file.parents[1]      # .../projects/DINOv3
    projects_dir = project_dir.parent       # .../projects
    mmseg_root = projects_dir.parent        # .../mmsegmentation

    for path in (mmseg_root, project_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _import_custom_modules():
    _add_import_paths()

    try:
        import projects.DINOv3.datasets  # noqa: F401
        print('[OK] Imported projects.DINOv3.datasets')
        return
    except Exception as exc:
        print(f'[WARN] Could not import projects.DINOv3.datasets: {exc}')

    try:
        import datasets  # noqa: F401
        print('[OK] Imported local datasets package')
        return
    except Exception as exc:
        raise RuntimeError(
            'Failed to import custom dataset modules. Make sure this file is under '
            'mmsegmentation/projects/DINOv3/datasets/.'
        ) from exc


def _make_fake_pastis(root: Path, split: str, num_samples: int = 4):
    split_dir = root / split
    image_dir = split_dir / 's2_images'
    image_dir.mkdir(parents=True, exist_ok=True)

    targets = torch.randint(low=0, high=19, size=(num_samples, 64, 64), dtype=torch.long)
    targets[0, :8, :8] = -1

    for idx in range(num_samples):
        img = torch.rand(12, 13, 64, 64)
        torch.save(img, image_dir / f'{idx}.pt')

    torch.save(targets, split_dir / 'targets.pt')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default=None, help='Path to pastis_dataset_64.')
    parser.add_argument('--split', default='pastis_r_train')
    parser.add_argument('--resize', nargs=2, type=int, default=None, metavar=('H', 'W'))
    parser.add_argument('--temporal-reduce', default='mean')
    parser.add_argument('--make-fake', action='store_true')
    parser.add_argument('--keep-minus-one-ignore', action='store_true',
                        help='Keep original -1 ignore index instead of mapping it to 255.')
    return parser.parse_args()


def main():
    args = parse_args()
    _import_custom_modules()

    from mmseg.registry import DATASETS

    tmp_dir = None
    if args.make_fake:
        tmp_dir = Path(tempfile.mkdtemp(prefix='fake_pastis_'))
        data_root = tmp_dir / 'pastis_dataset_64'
        _make_fake_pastis(data_root, args.split)
        print(f'[OK] Created fake dataset at: {data_root}')
    else:
        if args.data_root is None:
            raise ValueError('Please pass --data-root, or use --make-fake for a self-contained test.')
        data_root = Path(args.data_root)

    pipeline = [
        dict(
            type='LoadPastisSampleFromPT',
            temporal_reduce=args.temporal_reduce,
            source_ignore_index=-1,
            target_ignore_index=-1 if args.keep_minus_one_ignore else 255,
        ),
    ]

    if args.resize is not None:
        pipeline.append(dict(type='PastisResize', size=tuple(args.resize)))

    pipeline.append(dict(type='PastisPackSegInputs'))

    dataset_cfg = dict(
        type='PastisPtDataset',
        data_root=str(data_root),
        split=args.split,
        pipeline=pipeline,
    )

    dataset = DATASETS.build(dataset_cfg)
    print(f'[OK] Dataset built. length={len(dataset)}')

    sample = dataset[0]
    inputs = sample['inputs']
    data_sample = sample['data_samples']
    gt = data_sample.gt_sem_seg.data

    print(f'[OK] inputs shape: {tuple(inputs.shape)} dtype={inputs.dtype}')
    print(f'[OK] gt_sem_seg shape: {tuple(gt.shape)} dtype={gt.dtype}')
    print(f'[OK] metainfo: {data_sample.metainfo}')

    unique_values = torch.unique(gt)
    print(f'[OK] unique label values in sample 0: {unique_values[:30].tolist()}')

    if tmp_dir is not None:
        shutil.rmtree(tmp_dir)
        print('[OK] Removed fake dataset.')


if __name__ == '__main__':
    main()
