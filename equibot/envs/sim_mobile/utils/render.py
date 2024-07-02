"""
Utilities for rendering and visualization.

Authors: @yjy0625, @contactrika

"""

import numpy as np
import pybullet


def add_debug_region(sim, mn, mx, clr=(1, 0, 0), z=0.01):
    sim.addUserDebugLine([mn[0], mn[1], z], [mn[0], mx[1], z], clr)
    sim.addUserDebugLine([mn[0], mn[1], z], [mx[0], mn[1], z], clr)
    sim.addUserDebugLine([mn[0], mx[1], z], [mx[0], mx[1], z], clr)
    sim.addUserDebugLine([mx[0], mn[1], z], [mx[0], mx[1], z], clr)
