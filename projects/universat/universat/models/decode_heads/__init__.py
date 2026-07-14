"""UniverSat decode heads for MMSegmentation 1.x."""

from .universat_seg_head import UniverSatSegHead
from .universat_lp_head import UniverSatLinearProbeHead

__all__ = ["UniverSatSegHead", "UniverSatLinearProbeHead"]
