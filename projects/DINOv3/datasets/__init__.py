"""PASTIS dataset support for DINOv3 under MMSegmentation projects.

Expected location:
    mmsegmentation/projects/DINOv3/datasets/

The most important purpose of this file is to import all dataset / transform
classes so that MMEngine registries can see them after:

    custom_imports = dict(
        imports=['projects.DINOv3.datasets'],
        allow_failed_imports=False,
    )
"""

from .pastis_pt_dataset import PastisPtDataset
from .transforms import (
    LoadPastisSampleFromPT,
    PastisResize,
    PastisPackSegInputs,
)

__all__ = [
    'PastisPtDataset',
    'LoadPastisSampleFromPT',
    'PastisResize',
    'PastisPackSegInputs',
]
