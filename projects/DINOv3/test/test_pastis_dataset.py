import argparse
import sys
from pathlib import Path

import torch


def parse_int_list(text):
    if text is None or text == '' or text.lower() == 'none':
        return None

    return tuple(int(item.strip()) for item in text.split(',') if item.strip())


def add_python_paths(project_root, mmseg_root):
    for path in (mmseg_root, project_root):
        path = str(Path(path).resolve())
        if path not in sys.path:
            sys.path.insert(0, path)


def import_project_modules():
    try:
        import projects.DINOv3.datasets  # noqa: F401
        import mmseg.datasets.transforms  # noqa: F401
    except Exception as exc:
        raise ImportError(
            'Failed to import project dataset modules. Please run this script '
            'inside the MMSegmentation repository, or pass --mmseg-root and '
            '--project-root explicitly.'
        ) from exc


def build_dataset(args):
    from mmseg.registry import DATASETS

    loader_cfg = dict(
        type='LoadPastisSampleFromPT',
        temporal_mode=args.temporal_mode,
        time_index=args.time_index,
        band_indices=parse_int_list(args.band_indices),
        resize_size=None if args.resize_size is None else tuple(args.resize_size),
        ignore_index=args.ignore_index,
        target_ignore_index=args.target_ignore_index,
        to_float32=True,
    )

    pipeline = [
        loader_cfg,
        dict(
            type='PackSegInputs',
            meta_keys=(
                'img_path',
                'seg_map_path',
                'ori_shape',
                'img_shape',
                'pad_shape',
                'scale_factor',
                'reduce_zero_label',
                'sample_idx',
                'split',
                'num_channels',
            ),
        ),
    ]

    dataset_cfg = dict(
        type='PASTISDataset64',
        data_root=args.data_root,
        split=args.split,
        pipeline=pipeline,
        ignore_index=args.ignore_index,
    )

    print('[Test] Dataset config:')
    print(dataset_cfg)

    dataset = DATASETS.build(dataset_cfg)
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description='Test PASTISDataset64 loading for MMSegmentation.'
    )

    parser.add_argument(
        '--data-root',
        required=True,
        help='Path to pastis_dataset_64.'
    )

    parser.add_argument(
        '--split',
        default='train',
        choices=['train', 'val', 'test'],
        help='Dataset split to test.'
    )

    parser.add_argument(
        '--project-root',
        default=None,
        help=(
            'Path to mmsegmentation/projects/DINOv3. '
            'If omitted, infer from this file.'
        )
    )

    parser.add_argument(
        '--mmseg-root',
        default=None,
        help=(
            'Path to MMSegmentation root. '
            'If omitted, infer from projects/DINOv3/test.'
        )
    )

    parser.add_argument(
        '--temporal-mode',
        default='mean',
        choices=['mean', 'select', 'max', 'flatten'],
        help='How to reduce the 12-month temporal dimension.'
    )

    parser.add_argument(
        '--time-index',
        type=int,
        default=0,
        help='Month index used when --temporal-mode select.'
    )

    parser.add_argument(
        '--band-indices',
        default=None,
        help=(
            'Comma-separated band indices to keep after 13-band construction. '
            'Example: "3,2,1" keeps B4/B3/B2 as RGB-like input. '
            'Default keeps all 13 bands.'
        )
    )

    parser.add_argument(
        '--resize-size',
        type=int,
        nargs=2,
        default=None,
        metavar=('HEIGHT', 'WIDTH'),
        help='Optional resize output size, e.g. --resize-size 224 224.'
    )

    parser.add_argument(
        '--ignore-index',
        type=int,
        default=-1,
        help='Ignore label value in targets.pt.'
    )

    parser.add_argument(
        '--target-ignore-index',
        type=int,
        default=None,
        help=(
            'Optional value to replace ignore_index with, '
            'e.g. 255 for default MMSeg configs.'
        )
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        default=3,
        help='Number of samples to inspect.'
    )

    args = parser.parse_args()

    this_file = Path(__file__).resolve()

    if args.project_root is not None:
        project_root = Path(args.project_root).expanduser().resolve()
    else:
        project_root = this_file.parents[1]

    if args.mmseg_root is not None:
        mmseg_root = Path(args.mmseg_root).expanduser().resolve()
    else:
        if project_root.parent.name == 'projects':
            mmseg_root = project_root.parent.parent.resolve()
        else:
            mmseg_root = Path.cwd().resolve()

    print(f'[Test] project_root: {project_root}')
    print(f'[Test] mmseg_root: {mmseg_root}')
    print(f'[Test] data_root: {Path(args.data_root).expanduser().resolve()}')

    add_python_paths(project_root, mmseg_root)
    import_project_modules()

    dataset = build_dataset(args)

    print(f'[Test] Dataset length: {len(dataset)}')
    print(f'[Test] Metainfo classes: {len(dataset.metainfo.get("classes", []))}')

    num_samples = min(args.num_samples, len(dataset))

    for i in range(num_samples):
        item = dataset[i]

        inputs = item['inputs']
        data_sample = item['data_samples']
        gt = data_sample.gt_sem_seg.data

        print(f'\n[Test] sample {i}')
        print(f'  inputs shape: {tuple(inputs.shape)}, dtype: {inputs.dtype}')
        print(f'  gt shape: {tuple(gt.shape)}, dtype: {gt.dtype}')
        print(f'  img_path: {data_sample.metainfo.get("img_path")}')
        print(f'  sample_idx: {data_sample.metainfo.get("sample_idx")}')

        unique_labels = torch.unique(gt.cpu())

        if unique_labels.numel() > 30:
            shown = unique_labels[:30].tolist()
            print(
                f'  unique labels first 30: {shown} ... '
                f'total={unique_labels.numel()}'
            )
        else:
            print(f'  unique labels: {unique_labels.tolist()}')

    print('\n[Test] PASTIS dataset loading success.')


if __name__ == '__main__':
    main()
