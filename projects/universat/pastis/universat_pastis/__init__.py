"""UniverSat PASTIS-R downstream project."""

# Import the base project so its models are registered before PASTIS builds.
import projects.universat.universat  # noqa: F401

from .datasets import *  # noqa: F401,F403
from .transforms import *  # noqa: F401,F403
from .utils import *  # noqa: F401,F403
