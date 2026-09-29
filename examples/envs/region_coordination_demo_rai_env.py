#!/usr/bin/env python
"""RAI dual-Franka-arm scenario built specifically to showcase MOTION-DERIVED MINIMAL TEMPORAL
CONSTRAINTS for shared-region conflicts (see docs/minimal_temporal_constraints.md) -- distinct
from DurationConflictRaiEnvironment (which this reuses the scene layout of), whose scenario exists
to show what a WRONG PDDL action :duration causes when nothing protects a shared surface at all.

Same 2-block-to-1-shared-drop_pad layout, same reason it's a good fit (the two robots' action
chains share no object, so nothing but the region mutex can force them apart) -- but here
region_mutex_enabled defaults to being the point, not the counterexample. Reuses
DurationConflictRaiEnvironment's scene-building, Action/subgoal-sampling and compute_path
machinery unchanged (see that module for those); this subclass only exists to give the scenario
its own identity/docstring/CLI entry so it reads as "the minimal-temporal-constraints demo," not
a repurposed borrow of a scenario built for something else.

Run through the full pipeline (Tamer -> the real, untouched PlanSkeleton/dRRT* -> composite path):
    python main_rai.py --env_type exp_region_coordination_demo --num_robots 2 --num_objs 2 \\
        --use_pddl_planner --region_mutex_enabled --region_offsets_from_motion

Or see compare_region_fluents_demo.py at the repo root for the fully general per-(robot, region)
numeric-fluent mode (not the scalar approximation the full pipeline above is limited to -- see
that script's docstring for why), driven by this same environment's real measured geometry.
"""
from mm_drrt.utils.rai_motion_planner_utils import get_placement_gen
from examples.envs.duration_conflict_rai_env import DurationConflictRaiEnvironment

RegionCoordinationDemoCameraSetup = [(0, -1, 4), (0, 0, 0)]

# How far toward the right edge of drop_pad block0's (the RED block's) delivery placement is
# biased -- purely cosmetic, so the two delivered blocks land visibly apart instead of drop_pad's
# small footprint (see DurationConflictRaiEnvironment's PAD_SIZE comment) sometimes randomly
# placing them close together. See sample_placement()'s x_bias (mm_drrt/utils/rai_utils.py) for
# how this clamps back within the pad's actual valid margin rather than ever landing off it.
BLOCK0_PLACEMENT_X_BIAS = 0.03


class RegionCoordinationDemoRaiEnvironment(DurationConflictRaiEnvironment):
    def placement_sample(self, m_objs, f_obj, num_samples):
        if f_obj == 'drop_pad' and m_objs == self.m_objs[0]:
            placement_gen = get_placement_gen(self._C, x_bias=BLOCK0_PLACEMENT_X_BIAS)(m_objs, f_obj)
            return [next(placement_gen)[0] for _ in range(num_samples)]
        return super().placement_sample(m_objs, f_obj, num_samples)
