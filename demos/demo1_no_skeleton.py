#!/usr/bin/env python
"""Contribution 1: a temporal MR-TAMP framework that removes MM-dRRT's given-plan-skeleton
assumption by using an off-the-shelf PDDL 2.1 planner (Tamer, via unified-planning).

Scenario: two fixed Franka arms relaying two boxes in opposite directions across one table
(exp_two_robots_rai, 2-object crossing relay) -- neither arm reaches both ends, so every box has
to be handed over at table_mid. Original MM-dRRT needs the whole task plan written by hand: which
robot does which pick/place, in what order per robot, per object, and which cross-robot
handoffs to wait for. Here the only input is the PDDL problem (objects, initial state, goal), and
Tamer produces the timed plan; the skeleton MM-dRRT needs is derived from it. Then the SAME,
unchanged PlanSkeleton/dRRT* refinement turns it into motion. Finally the goal changes (a single
box relay) and nothing has to be rewritten.

Usage:
    python demos/demo1_no_skeleton.py            # GUI walkthrough
    python demos/demo1_no_skeleton.py --no_gui   # headless
"""
from _walkthrough import demo_args, Walkthrough, pipeline_opt, setup_scene, close_scene, gantt, \
    schedule_rows, replay_sync, table, robot_index

from mm_drrt.pipeline_rai import solve_task_plan, refine
from mm_drrt.utils.pddl_parser import extract_sequential_constraints


def fmt_action(env, a):
    a_type, robot, m_obj, f_from, f_to = a
    r = f"r{robot_index(env, robot)}"
    if a_type == 'transit':
        return f"{r} pick {m_obj} from {f_to}"
    if a_type == 'transfer':
        return f"{r} place {m_obj} on {f_to}"
    return f"{r} {a_type}"


def show_skeleton(w, env, plan, action_orders, obj_orders, constraints):
    table(['action', 'meaning'], [(n, fmt_action(env, a)) for n, a in plan.items()])
    print()
    for robot, names in action_orders.items():
        print(f"  r{robot_index(env, robot)} action order: {' -> '.join(names)}")
    for obj, names in obj_orders.items():
        print(f"  {obj} placements in order: {' -> '.join(names)}")
    for c in constraints:
        print(f"  cross-robot: {c['pre']} ({fmt_action(env, plan[c['pre']])}) must finish before "
              f"{c['post']} ({fmt_action(env, plan[c['post']])}) starts")


def same_skeleton(manual, derived):
    """Compares two skeletons up to action naming: the same pick/place actions, the same order
    per robot, and the same cross-robot handoffs."""
    def canon(plan, action_orders, constraints):
        key = lambda n: (plan[n][0], plan[n][1], plan[n][2], plan[n][4])
        orders = {r: tuple(key(n) for n in names) for r, names in action_orders.items()}
        handoffs = {(key(c['pre']), key(c['post'])) for c in constraints}
        return sorted(map(key, plan), key=repr), orders, handoffs
    m_plan, m_orders, _, m_cons = manual
    d_plan, d_orders, _, d_cons = derived
    return canon(m_plan, m_orders, m_cons) == canon(d_plan, d_orders, d_cons)


def show_pddl(env):
    objects, init, goal = env.create_pddl_problem()
    robots = list(env.robots.values())
    name = lambda x: f"r{robots.index(x)}" if x in robots else str(x)
    for t, objs in objects.items():
        print(f"  {t:12s}: {', '.join(name(o) for o in objs)}")
    print("  init:  " + '\n         '.join(f"({p} {' '.join(name(a) for a in args)})" for p, *args in init))
    print("  goal:  " + '\n         '.join(f"({p} {' '.join(name(a) for a in args)})" for p, *args in goal))


def run_scenario(w, args, num_objs, label):
    opt = pipeline_opt(not args.no_gui, args.seed, '--env_type', 'exp_two_robots_rai', '--num_robots', '2',
                       '--num_objs', str(num_objs), '--use_pddl_planner', '--no_offsets_cache')
    C, env = setup_scene(opt)
    try:
        if num_objs == 2:
            w.step("What original MM-dRRT needs: a hand-written plan skeleton",
                   """MM-dRRT refines a GIVEN task plan. For this crossing relay someone has to write
                   every pick and place, each robot's action order, each box's placement order, and
                   which cross-robot handoffs to wait for -- this is the environment's
                   create_plan_order_constraints():""")
            show_skeleton(w, env, *env.create_plan_order_constraints())
            w.pause()

        w.step(f"The only input now: a PDDL 2.1 problem ({label})",
               """Objects, the initial state and the goal -- no actions, no orders, no handoffs. The
               domain (durative transit/transfer actions) is generic and shared by every
               environment.""")
        show_pddl(env)
        w.pause()

        w.step("Tamer (off-the-shelf PDDL 2.1 temporal planner) solves it",
               """Tamer returns a TIMED plan. The parser derives everything MM-dRRT used to be
               given: per-robot action orders, per-object orders, and the cross-robot handoffs
               (robot A places a box that robot B later picks up).""")
        plan, action_orders, obj_orders, constraints, used, schedule, _, _ = solve_task_plan(env, opt, None)
        print(f"  planner: {used}\n")
        handoffs = [c for c in constraints if plan[c['pre']][2] == plan[c['post']][2]]
        show_skeleton(w, env, plan, action_orders, obj_orders, handoffs)
        if schedule:
            print()
            gantt(schedule_rows(env, schedule, occupancy={}), title="  Tamer's timed plan:")
        if num_objs == 2:
            same = same_skeleton(env.create_plan_order_constraints(), (plan, action_orders, obj_orders, handoffs))
            print(f"\n  Same skeleton as the hand-written one (actions, per-robot orders, handoffs): "
                  f"{'yes' if same else 'no -- a different valid plan'}")
        w.pause()

        w.step("The SAME, unchanged MM-dRRT refinement turns it into motion",
               """PlanSkeleton (placements, grasps/IK subgoals, per-robot roadmaps) and the dRRT*
               composite search run exactly as they do on a hand-written skeleton -- they never see
               where the plan came from.""")
        composite_path, secs = refine(env, opt, plan, obj_orders, constraints)
        print(f"  refinement took {secs:.1f}s; composite path has {len(composite_path)} nodes")
        if w.use_gui:
            w.say("Replaying in the viewer...")
            replay_sync(C, env, plan, composite_path, w.use_gui)
        w.pause()
        return plan
    finally:
        close_scene(C)


def main():
    args = demo_args(__doc__.split('\n')[1])
    w = Walkthrough("Contribution 1 -- no given plan skeleton: PDDL 2.1 planning + MM-dRRT refinement",
                    not args.no_gui)
    w.step("Scenario",
           """Two fixed Franka arms. Each reaches only its own end of the table and the shared
           middle, so every box crossing the table has to be handed over at table_mid. Two boxes
           travel in opposite directions.""")
    run_scenario(w, args, 2, 'two boxes, crossing relay')
    w.step("Change the goal, write nothing new",
           """A different task -- one box relayed from table_start to table_end -- is just a
           different PDDL goal. No new skeleton is written; the same pipeline plans and refines
           it.""")
    run_scenario(w, args, 1, 'one box relay')
    w.step("Summary",
           """The task plan, each robot's action order, the object orders and the cross-robot
           handoffs were all produced by an off-the-shelf PDDL 2.1 planner from the goal alone,
           and refined by the unchanged MM-dRRT motion layer: the given-skeleton assumption is
           gone.""")


if __name__ == '__main__':
    main()
