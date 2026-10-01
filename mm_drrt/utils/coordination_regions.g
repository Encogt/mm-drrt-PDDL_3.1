## Standalone RAI scene defining the coordination regions used by
## examples/envs/region_coordination_demo_rai_env.py / duration_conflict_rai_env.py, as plain,
## hand-authored, real coordinates -- an alternative to region_volume_from_frame()'s default
## (read live from whatever Config an actual environment built) for when a region's bounding
## volume is needed WITHOUT constructing/connecting to a real environment at all. Loaded via
## region_volume_from_g_file() in coordination_regions.py, which reuses region_volume_from_frame()
## against a throwaway Config this file is loaded into -- so it stays the single source of truth
## for how a bounding volume is derived from a frame; only WHERE the frame comes from differs.
##
## Coordinates match examples/envs/duration_conflict_rai_env.py's ZONE_X/PAD_X/SURFACE_Y/
## SURFACE_Z/ZONE_SIZE/PAD_SIZE exactly -- these are the real positions/sizes that module's
## _create_problem() builds at runtime, not independent placeholders. If those constants ever
## change, this file needs to change with them (there is no automated link between the two).

world: {}

zone_left (world): { Q: "t(-0.8 0.15 0.6)", shape: box, size: [0.3, 0.3, 0.02], color: [0.5, 0.5, 0.9] }
zone_right (world): { Q: "t(0.8 0.15 0.6)", shape: box, size: [0.3, 0.3, 0.02], color: [0.5, 0.5, 0.9] }
drop_pad (world): { Q: "t(0.0 0.15 0.6)", shape: box, size: [0.20, 0.10, 0.02], color: [0.9, 0.7, 0.3] }
