"""Host-side estimator for the CAD Mouse MK2 magnetic knob.

Nine magnetic field readings in, six degrees of freedom out. The pieces:

``geometry``  fixed board and knob geometry, and which parts of it are trusted
``magnet``    field of one cylinder: exact reference, interpolation table, dipole
``params``    what calibration is allowed to move, and what it is not
``model``     the measurement function and its Jacobian
``dataset``   loading a recorded session

The calibration fit and the filter build on these and never run on the device;
the measurement function does, so ``magnet`` and ``model`` are written to be
ported rather than merely to be correct here.
"""

from .geometry import CHANNEL_NAMES, COUNTS_PER_MT, SENSOR_POS
from .magnet import FieldTable, build_table, field_axisym_exact, sample, sample_scalar
from .model import forward, forward_and_jac, solve_pose
from .params import CalibParams

__all__ = [
    "CHANNEL_NAMES",
    "COUNTS_PER_MT",
    "SENSOR_POS",
    "FieldTable",
    "build_table",
    "field_axisym_exact",
    "sample",
    "sample_scalar",
    "forward",
    "forward_and_jac",
    "solve_pose",
    "CalibParams",
]
