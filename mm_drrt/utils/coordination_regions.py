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


# Real, hand-authored coordinates for the regions region_volume_from_g_file() below can load --
# matches examples/envs/duration_conflict_rai_env.py's ZONE_X/PAD_X/SURFACE_Y/SURFACE_Z/ZONE_SIZE/
# PAD_SIZE exactly (see that file for why these particular numbers). Not the default path --
# region_volume_from_frame() against a real environment's live Config always matches whatever
# that environment actually built, this doesn't -- but a legitimate alternative when a region's
# volume is needed without constructing/connecting to a real environment at all.
DEFAULT_G_FILE = __file__.rsplit('.py', 1)[0] + '.g'


def region_volume_from_g_file(frame_name, g_file=DEFAULT_G_FILE, height_margin=DEFAULT_HEIGHT_MARGIN):
    """Alternative to region_volume_from_frame() for when no live environment Config is
    available/wanted: loads frame_name's definition from a standalone .g scene file (default:
    coordination_regions.g, next to this module) into a throwaway ry.Config, then reads it the
    exact same way region_volume_from_frame() does -- so that function stays the single source of
    truth for how a bounding volume is derived from a frame; only WHERE the frame comes from
    differs here."""
    import robotic as ry
    C = ry.Config()
    C.addFile(g_file)
    return region_volume_from_frame(C, frame_name, height_margin=height_margin)
