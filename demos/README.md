# Demos

GUI walkthroughs for the four contributions. Each one opens the RAI viewer, explains every step in
the terminal and waits for Enter before going on. Add `--no_gui` to run any of them headlessly:
the narration then advances on its own and nothing is replayed.

| Demo | Contribution | Scene |
|---|---|---|
| `demo1_no_skeleton.py` | Temporal MR-TAMP without a given plan skeleton: Tamer (PDDL 2.1) plans from the goal alone, and the unchanged MM-dRRT refines the result | `exp_two_robots_rai`: two-box crossing relay, then a one-box relay |
| `demo2_motion_intervals.py` | Motion-derived coordination intervals [α, β] and durations, re-measured on the executed trajectory and repaired | `exp_region_coordination_demo` |
| `demo3_least_commitment.py` | Least-commitment conflict resolution vs. no constraint and full-action serialization | `exp_region_coordination_demo` |
| `demo4_async_scaling.py` | Asynchronous execution and re-timing vs. lock-step execution and full serialization, for N arms | `exp_round_table` (N Franka arms around one shared pad) |

```
python demos/demo1_no_skeleton.py
python demos/demo2_motion_intervals.py
python demos/demo3_least_commitment.py
python demos/demo4_async_scaling.py --num_robots 3        # --speed 0.5 slows the asynchronous replay down
python demos/demo4_async_scaling.py --sweep 2,3           # headless scaling table
```

Times are seconds of velocity-limited motion (Franka joint velocity limits), not wall-clock time.
Each demo runs in under 5 minutes headless. At N = 4 refinement takes one to two minutes: dRRT*
gets at most 2 attempts of 60 s, each with a fresh seed.

Measured with demo 4 (seed 0, headless). Makespans are in seconds; the asynchronous variants all
execute the same refined paths.

| N | lock-step | async, full mutex | async, minimal | async, motion conflict |
|---|---|---|---|---|
| 2 | 8.07 | 8.55 | 8.07 | 8.07 |
| 3 | 8.23 | 11.25 | 7.65 | 7.65 |
| 4 | 9.00 | 14.67 | 9.58 | 10.13 |

The minimal intervals always beat full serialization on the same paths. Compared with lock-step,
they help at N = 3 (-7%) but not at N = 4 (+6.5%). In this setup dRRT* itself is given no pad
precedence, and its lock-step path only has to avoid real collisions. The asynchronous re-timing
instead keeps whole occupancy or conflict windows apart, so it is the more conservative of the two.
