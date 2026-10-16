#!/usr/bin/env python
"""Contribution 2: a motion-derived representation of coordination-critical temporal intervals.

Each high-level action i carries O_i = {(R_k, [alpha_ik, beta_ik])}: the sub-interval of the action
during which robot i actually occupies coordination region R_k. Instead of declaring these by
hand, they are measured from real IK/motion-planned trajectories: the gripper's forward kinematics
against the region's volume, timed under the arm's joint velocity limits. The walkthrough:

  1. animates representative trajectories for every (robot, region) pair, with the region volume
     drawn, and reports when the gripper enters and leaves it;
  2. shows why the interval must be measured in TIME, not waypoint indices (re-interpolating a path
     moves index fractions but not times);
  3. shows that two robots get different intervals for the same region;
  4. feeds them to Tamer, refines, and closes the loop: the executed dRRT* trajectory is measured
     the same way, a violation widens the intervals/durations, and the plan is re-solved until
     planned and executed timing agree.

Usage:
    python demos/demo2.py            # GUI walkthrough
    python demos/demo2.py --no_gui   # headless
"""
import random
import time

import numpy as np

from _walkthrough import demo_args, Walkthrough, pipeline_opt, setup_scene, close_scene, show_region, \
    replay_sync, table

from mm_drrt import pipeline_rai
from mm_drrt.utils import rai_utils as ru
from mm_drrt.utils.coordination_regions import point_in_region
from mm_drrt.utils.motion_timing import velocity_limits, path_timestamps, trajectory_region_times
from mm_drrt.utils.rai_motion_planner_utils import sample_region_offsets, trajectory_region_fractions, \
    gripper_positions_along_path

DECLARED_DURATION = 10.0


def animate(C, robot, path, region, use_gui, pause=0.03):
    """Plays one arm path and prints the waypoints where the gripper enters/leaves `region`."""
    arm_joints, gripper_frame, _, _ = ru._spec_of(robot)
    positions = gripper_positions_along_path(robot, arm_joints, gripper_frame, path)
    inside_prev = False
    for i, (q, p) in enumerate(zip(path, positions)):
        inside = point_in_region(p, region)
        if inside != inside_prev:
            print(f"    waypoint {i:3d}/{len(path) - 1}: gripper {'ENTERS' if inside else 'LEAVES'} the region")
            inside_prev = inside
        if use_gui:
            ru.set_joint_positions(robot, arm_joints, list(q)[-len(arm_joints):])
            C.view(False)
            time.sleep(pause)


def densify_before(path, k, factor=4):
    """`path` with `factor` - 1 extra interpolated waypoints inserted between each pair before k --
    the same motion, just sampled more finely in its first part."""
    out = []
    for i in range(len(path) - 1):
        a, b = np.asarray(path[i], dtype=float), np.asarray(path[i + 1], dtype=float)
        n = factor if i < k else 1
        out += [list(a + (b - a) * j / n) for j in range(n)]
    return out + [list(path[-1])]


def main():
    args = demo_args(__doc__.split('\n')[1])
    w = Walkthrough("Contribution 2 -- motion-derived coordination-critical intervals", not args.no_gui)
    opt = pipeline_opt(w.use_gui, args.seed, '--env_type', 'exp_region_coordination_demo', '--num_robots', '2',
                       '--num_objs', '2', '--use_pddl_planner', '--region_mutex_enabled',
                       '--region_offsets_from_motion', '--durations_from_motion', '--region_offset_mode', 'two',
                       '--repair_strategy', 'resolve', '--max_resolve_attempts', '2', '--no_offsets_cache')
    # Step 5 runs the arms slower than the planner's model (see there); everything before it
    # measures with the model itself.
    exec_scale = 0.6
    C, env = setup_scene(opt)
    try:
        w.step("Scenario and what is being measured",
               """Two Franka arms each pick a block from their own zone and place it on ONE shared
               drop_pad. A pick 'occupies' its zone and a place 'occupies' the pad only for part of
               the action -- from the moment the gripper enters the region's volume until it leaves.
               That sub-interval [alpha, beta] is what other robots must not overlap with. The
               translucent boxes in the viewer are those region volumes.""")
        regions = {}
        for frame, color in ((env.f_objs[0], (0.2, 0.4, 1.0)), (env.f_objs[1], (0.2, 0.4, 1.0)),
                             (env.f_objs[2], (1.0, 0.5, 0.1))):
            regions[frame] = show_region(C, frame, color=color, use_gui=w.use_gui)

        w.step("Measuring alpha/beta from representative trajectories",
               f"""For every (robot, region) pair: sample a grasp/placement, plan the approach and
               retreat with the same IK and roadmap primitives the real refinement uses, then run
               forward kinematics along the path. alpha/beta are reported two ways: as WAYPOINT-INDEX
               fractions scaled to a declared {DECLARED_DURATION:.0f}s action (the earlier method),
               and as TIME under the Franka joint velocity limits, where the duration itself also
               comes from the motion. The viewer previews each arm motion only; blocks are not
               carried in these previews.""")
        rows, measured, paths = [], {'transit': [], 'transfer': []}, {}
        for r in range(2):
            robot = env.robots[r]
            arm_joints, gripper_frame, _, _ = ru._spec_of(robot)
            vlim = velocity_limits(arm_joints)
            for action_type, frame in (('transit', env.f_objs[r]), ('transfer', env.f_objs[2])):
                saved = env.save_world()
                out = sample_region_offsets(robot, env._arm, env._grasp_type, env.m_objs[r], frame, action_type,
                                            None, collision_objs=env.fixed_obstacles, velocity_limits=vlim,
                                            return_path=True)
                if out is None:
                    env.restore_world(saved)
                    print(f"  r{r} {action_type}: no valid sample")
                    continue
                (t_in, t_out, dur), path = out
                fr = trajectory_region_fractions(robot, arm_joints, gripper_frame, path, regions[frame])
                print(f"\n  r{r} {action_type} ({'pick from' if action_type == 'transit' else 'place on'} {frame}):")
                # Sampling moved the block to its sampled pose; put the scene back first so the
                # preview only moves the arm instead of showing the block jump around.
                env.restore_world(saved)
                animate(C, robot, path, regions[frame], w.use_gui)
                rows.append((f"r{r}", action_type, frame, len(path),
                             f"[{fr[0] * DECLARED_DURATION:.2f}, {fr[1] * DECLARED_DURATION:.2f}]",
                             f"[{t_in:.2f}, {t_out:.2f}]", f"{dur:.2f}"))
                measured[action_type].append((t_in, t_out, dur))
                paths[(r, action_type)] = (robot, path, regions[frame])
        print()
        table(['robot', 'action', 'region', 'waypoints', 'index [a,b] (of 10s)', 'time [a,b] (s)', 'duration (s)'], rows)
        w.pause()

        w.step("Why time, not waypoint indices",
               """A path's waypoints are not evenly spaced in time: interpolation density varies
               (inside one dRRT* composite node the two arms routinely carry different waypoint
               counts). Below, the SAME transfer path is re-sampled 4x more finely before the
               gripper reaches the pad. The motion is identical, but the index-based interval
               jumps; the velocity-limited times change only by how finely the boundary crossing
               itself is sampled.""")
        robot, path, region = paths[(0, 'transfer')]
        arm_joints, gripper_frame, _, _ = ru._spec_of(robot)
        vlim = velocity_limits(arm_joints)
        k = trajectory_region_fractions(robot, arm_joints, gripper_frame, path, region)
        k_idx = int(round(k[0] * (len(path) - 1)))
        dense = densify_before(path, k_idx)
        cmp_rows = []
        for label, p in (('original', path), ('densified', dense)):
            fr = trajectory_region_fractions(robot, arm_joints, gripper_frame, p, region)
            t = trajectory_region_times(robot, arm_joints, gripper_frame, p, region, vlim)
            cmp_rows.append((label, len(p), f"[{fr[0] * DECLARED_DURATION:.2f}, {fr[1] * DECLARED_DURATION:.2f}]",
                             f"[{t[0]:.3f}, {t[1]:.3f}]", f"{t[2]:.3f}"))
        table(['path', 'waypoints', 'index [a,b] (of 10s)', 'time [a,b] (s)', 'duration (s)'], cmp_rows)
        w.pause()

        w.step("Per-(robot, region) intervals differ",
               """The two arms approach the same drop_pad from opposite sides with different
               kinematics, so their measured [alpha, beta] and durations differ. The PDDL domain
               can carry them per (robot, region) as numeric fluents; the full pipeline below uses
               the conservative per-action-type union (earliest entry, latest exit, longest
               duration), which never under-protects a region.""")
        planned = {t: (min(a for a, b, d in v), max(b for a, b, d in v), max(d for a, b, d in v))
                   for t, v in measured.items() if v}
        for t, (a, b, d) in planned.items():
            print(f"  {t:8s}: occupancy [{a:.3f}, {b:.3f}] s of a {d:.3f} s action")
        w.pause()

        w.step("Plan with them, refine, and measure the EXECUTED trajectory",
               """Tamer schedules with these intervals and durations; PlanSkeleton/dRRT* refines the
               plan. The executed per-robot trajectories are then split into actions and measured
               exactly the same way. A representative sample is not the trajectory dRRT* actually
               produces, so planned and executed timing can disagree; when the executed motion
               leaves its planned window or takes longer (beyond a 0.5 s tolerance), the
               intervals/durations are widened and the plan is re-solved (up to 2 times).

               To make the disagreement real and visible, the arms here EXECUTE at 60% of the joint
               speed the planner modelled -- a model mismatch, as when a real controller runs
               slower than the nominal limits. Every executed motion then takes longer and leaves
               its region later than planned: the planned windows are genuinely too narrow, and
               the loop has to fix them.""")
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        opt.execution_velocity_scale = exec_scale
        result = pipeline_rai.run_pipeline(env, opt, planned, 'demo2')
        print(f"\n  attempts: {result.attempts}")
        if result.timeline:
            exec_rows = []
            for n, a in sorted(result.timeline.items()):
                ent = f"[{a['entry']:.3f}, {a['exit']:.3f}]" if a['entry'] is not None else 'never inside'
                exec_rows.append((n, f"r{a['robot_index']}", a['type'], a['region'], ent, f"{a['duration']:.3f}"))
            print("  executed trajectory, per action:")
            table(['action', 'robot', 'type', 'region', 'time [a,b] (s)', 'duration (s)'], exec_rows)
            print(f"  final planned durations: { {t: round(float(d), 3) for t, d in result.durations.items()} }")
        w.pause()

        if w.use_gui:
            w.step("Replay", "The refined plan, replayed in the viewer.")
            replay_sync(C, env, result.plan, result.composite_path, w.use_gui, result.action_orders)

        w.step("Summary",
               """Coordination-critical intervals [alpha, beta] and action durations come from the
               motion itself -- forward kinematics against region volumes, timed by joint velocity
               limits -- per robot and per region, and are re-measured on the executed trajectory to
               keep the schedule consistent with what the robots really do.""")
    finally:
        close_scene(C)


if __name__ == '__main__':
    main()
