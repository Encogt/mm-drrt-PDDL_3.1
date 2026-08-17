"""Deliberately naive multi-robot execution scheduler, built to demonstrate what a wrong PDDL
action :duration causes when an executor trusts it directly for cross-robot timing instead of
using dRRT*'s real inter-robot collision checking.

This is an INCOMPLETE stand-in for mm_drrt.planner.rai_task_planner.PlanSkeleton.plan_refinement's
Step 4 (composite_path_computation / assign_order_constraints, in rai_task_planner_utils.py /
rai_drrt_star.py), which DOES check get_inter_robots_collision_fn during its dRRT* tree growth
and would never let two robots' arms actually occupy colliding configurations, regardless of
what any PDDL :duration says. Steps 1-3 (placement refinement, subgoal refinement, individual
per-robot path computation) are reused UNCHANGED here -- they're pure single-robot geometry and
never reason about the other robot. Only Step 4 is replaced, with a common real-world
simplification some lightweight multi-robot systems actually use: "wait out the other robot's
DECLARED action duration, then go" -- instead of checking whether the shared workspace is
genuinely clear.

Intended for mm_drrt.examples.envs.duration_conflict_rai_env.DurationConflictRaiEnvironment: two
robots independently delivering their own block to one shared drop_pad, with no same-object
handoff linking them (see that module's docstring for why that matters).
"""
import time

from mm_drrt.planner.rai_task_planner import PlanSkeleton
from mm_drrt.utils.rai_task_planner_utils import subgoal_refinement, individual_path_computation
from mm_drrt.utils.rai_motion_planner_utils import get_inter_robots_collision_fn


def _extract_path(roadmap):
    """Every roadmap object returned by env.compute_path (a DegreePRM, see mm_drrt/planner/prm.py)
    records the exact (start, goal) it was solved for as .initial_conf/.final_conf. Calling it
    again with those same endpoints re-runs the (now-fixed) graph search and retraces the
    concrete waypoint path -- the same mechanism get_arm_motion_fn already uses internally
    (rai_motion_planner_utils.py) to build/validate its own roadmap while planning."""
    path = roadmap(roadmap.initial_conf, roadmap.final_conf)
    return path if path else [roadmap.final_conf]


def _robot_action_paths(roadmaps_for_robot):
    """roadmaps_for_robot: a flat [base, approach, retrieval] triple per action (env.compute_path's
    convention -- see get_trivial_roadmap's docstring for why even a fixed-arm env still
    contributes a placeholder base roadmap). Returns one flat approach+retrieval waypoint path
    per action, in action order (here: [transit_path, transfer_path])."""
    paths = []
    for i in range(0, len(roadmaps_for_robot), 3):
        _, approach_rm, retrieval_rm = roadmaps_for_robot[i:i + 3]
        paths.append(_extract_path(approach_rm) + _extract_path(retrieval_rm))
    return paths


def _sample_at(segments, f):
    """segments: ordered [(start_frame, path), ...] for one robot. For any frame in the gap
    after a segment finishes but before the next one's start_frame, holds at that segment's
    LAST conf -- this is the "idling, waiting out the declared duration" behavior the naive
    scheduler models (as opposed to dRRT*, which would instead actively check whether it's safe
    to proceed)."""
    current = segments[0][1][0]
    for start, path in segments:
        if f < start:
            return current
        if f < start + len(path):
            return path[f - start]
        current = path[-1]
    return current


def _refine_plan(env, plan, obj_orders, init_order_constraints, num_base_samples, use_debug, max_attempts=5):
    """Steps 1-2 of PlanSkeleton.plan_refinement (rai_task_planner.py), reused unchanged: pure
    single-robot placement/grasp geometry, no cross-robot reasoning at all. Retries with a fresh
    PlanSkeleton (full re-sample) rather than plan_refinement's own in-place retry loop, since
    this demo only needs to succeed once, not efficiently."""
    for attempt in range(max_attempts):
        skeleton = PlanSkeleton(env, plan, obj_orders, init_order_constraints, num_samples=num_base_samples,
                                use_debug=use_debug)
        for f in skeleton.f_objs:
            while True:
                if skeleton.f_objs[f].is_refine_placements():
                    break
        if subgoal_refinement(skeleton.robot_plans, env, skeleton.obj_orders, skeleton.actions, use_debug):
            return skeleton
        print(f"Naive duration demo: subgoal refinement failed (attempt {attempt + 1}/{max_attempts}), retrying...")
    raise RuntimeError("Naive duration demo: subgoal refinement did not converge")


def run_naive_duration_demo(env, C, transit_duration, transfer_duration,
                            num_base_samples=50, num_arm_samples=20, use_debug=False, pause_time=0.02):
    """Solves env.create_plan_order_constraints()'s plan via Steps 1-3 of
    PlanSkeleton.plan_refinement (unmodified), then plays both robots back on a naive timeline:
    robot 1's transfer is gated to start `transfer_duration` frames after robot 0's transfer
    begins -- trusting the DECLARED duration, not robot 0's real measured motion length --
    instead of dRRT*'s real inter-robot collision check (deliberately skipped here; see the
    module docstring). transit_duration is accepted for CLI/Tamer-duration parity but doesn't
    gate anything in this scenario -- each robot's own transit only touches its own zone.

    Returns (collision_frames, total_frames). collision_frames is the list of played-back frame
    indices where the two robots' arms were actually found in collision
    (get_inter_robots_collision_fn) -- the visible effect of a transfer_duration that undershoots
    the real motion time.
    """
    plan, _action_orders, obj_orders, init_order_constraints = env.create_plan_order_constraints()
    skeleton = _refine_plan(env, plan, obj_orders, init_order_constraints, num_base_samples, use_debug)

    roadmaps, heuristic_vals = [], []
    ok, roadmaps, heuristic_vals = individual_path_computation(
        skeleton.robot_plans, env, num_base_samples, num_arm_samples, roadmaps, heuristic_vals, use_debug)
    if not ok:
        raise RuntimeError("Naive duration demo: per-robot path computation failed")

    robots = list(skeleton.robot_plans.keys())
    joints = env.get_joints(robots)
    (r0_transit, r0_transfer), (r1_transit, r1_transfer) = [_robot_action_paths(r) for r in roadmaps]

    r0_transfer_start = len(r0_transit)
    r1_transfer_start = len(r0_transit) + int(round(transfer_duration))
    segments = [
        [(0, r0_transit), (r0_transfer_start, r0_transfer)],
        [(0, r1_transit), (r1_transfer_start, r1_transfer)],
    ]
    total_frames = max(r0_transfer_start + len(r0_transfer), r1_transfer_start + len(r1_transfer))

    print(f"Naive duration demo: declared transfer_duration={transfer_duration} frames "
          f"(transit_duration={transit_duration}, unused by this scenario) vs robot 0's real "
          f"transfer motion = {len(r0_transfer)} frames")

    inter_collision_fn = get_inter_robots_collision_fn(robots, joints, num_robots=len(robots))
    collision_frames = []
    for f in range(total_frames):
        q = [_sample_at(segments[i], f) for i in range(len(robots))]
        if inter_collision_fn(q, mode='boolean'):
            collision_frames.append(f)
        C.view(False)
        time.sleep(pause_time)

    if collision_frames:
        print(f"Naive duration demo: inter-robot collision detected at {len(collision_frames)}/{total_frames} "
              f"frames (first at frame {collision_frames[0]}) -- the declared transfer_duration "
              f"undershot robot 0's real retreat motion, so robot 1 started into the shared "
              f"workspace before robot 0 had actually cleared it.")
    else:
        print(f"Naive duration demo: no inter-robot collision across {total_frames} frames -- the "
              f"declared transfer_duration covered robot 0's real retreat motion.")
    return collision_frames, total_frames
