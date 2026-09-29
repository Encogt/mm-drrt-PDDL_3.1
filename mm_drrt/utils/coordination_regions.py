"""Concrete bounding volumes for PDDL region-conflict coordination.

A "region" here is a fixed-obj's real, live geometry in the RAI scene (a table/zone/pad frame),
read directly from the ry.Config rather than duplicated as constants -- so a region's bounding
volume always matches whatever the environment actually built (see e.g.
examples/envs/duration_conflict_rai_env.py's ZONE_SIZE/PAD_SIZE/create_box calls).

The box a region's frame is drawn with (its footprint) is not itself the contested volume: what
robots actually contest is the airspace ABOVE that footprint, where a gripper/held-object
descends into and retreats from it. region_volume() extrudes the frame's footprint upward by
`height_margin` to get that interaction column -- this is the bounding volume
trajectory_region_fractions() (mm_drrt/utils/rai_motion_planner_utils.py) tests a path against to
find when it actually enters/exits region k.
"""

from collections import namedtuple

# How far above a region's own top surface the contested airspace extends -- generous enough to
# cover a top-grasp approach/retreat (the gripper is well above the surface for most of that
# motion), tight enough not to swallow the robot's whole workspace.
DEFAULT_HEIGHT_MARGIN = 0.30

RegionVolume = namedtuple('RegionVolume', ['center', 'half_extents'])


def region_volume_from_frame(C, frame_name, height_margin=DEFAULT_HEIGHT_MARGIN):
    """Builds the region's interaction-column bounding volume from the frame's own live geometry.
    Assumes the frame is an axis-aligned box (true for every table/zone/pad frame in this repo --
    they're all created via create_box()/create_table() with no rotation set)."""
    frame = C.getFrame(frame_name)
    position = frame.getPosition()
    size = frame.getSize()  # [dx, dy, dz] full extents for a ry.ST.box

    half_extents = (size[0] / 2.0, size[1] / 2.0, height_margin / 2.0)
    # Column sits on top of the frame's own top face and extends upward by height_margin.
    center = (position[0], position[1], position[2] + size[2] / 2.0 + height_margin / 2.0)
    return RegionVolume(center=center, half_extents=half_extents)


def point_in_region(point, region):
    return all(abs(point[i] - region.center[i]) <= region.half_extents[i] for i in range(3))
