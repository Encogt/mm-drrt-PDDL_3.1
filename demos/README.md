# Demos

GUI walkthroughs for the four contributions. Each one opens the RAI viewer, explains every step in
the terminal and waits for Enter before going on. Add `--no_gui` to run any of them headlessly:
the narration then advances on its own and nothing is replayed.

| Demo | Contribution | Scene |
|---|---|---|
| `demo1.py` | Temporal MR-TAMP without a given plan skeleton: Tamer (PDDL 2.1) plans from the goal alone, and the unchanged MM-dRRT refines the result | `exp_two_robots_rai`: two-box crossing relay, then a one-box relay |
| `demo2.py` | Motion-derived coordination intervals [α, β] and durations, re-measured on the executed trajectory and repaired | `exp_region_coordination_demo` |
| `demo3.py` | Least-commitment conflict resolution vs. no constraint and full-action serialization | `exp_region_coordination_demo` |
| `demo4.py` | Asynchronous execution and re-timing vs. lock-step execution and full serialization, for N arms | `exp_round_table` (N Franka arms around one shared pad) |

```
python demos/demo1.py
python demos/demo2.py
python demos/demo3.py
python demos/demo4.py --num_robots 3        # --speed 0.5 slows the asynchronous replay down
python demos/demo4.py --sweep 2,3           # headless scaling table
```

Times are seconds of velocity-limited motion (Franka joint velocity limits), not wall-clock time.
Each demo runs in under 5 minutes headless. dRRT* gets at most 2 attempts of 60 s, each with a
fresh seed.

Measured with demo 4 (seed 0, headless). Makespans are in seconds; the variants all execute the
same refined paths.

| N | lock-step | async, full mutex | async, minimal | async, motion conflict |
|---|---|---|---|---|
| 2 | 4.81 | 6.20 | 4.81 | 4.81 |
| 3 | 6.41 | 9.08 | 5.57 | 5.57 |
| 4 | 8.03 | 12.06 | 7.78 | 7.78 |

The minimal intervals beat full serialization by 22-39%, and lock-step by 0% at N = 2, 13.2% at
N = 3 and 3.1% at N = 4. These are single runs. Tamer can name the plan's actions differently
between runs, which changes the random draws downstream, so the numbers vary from run to run.

What the replays show:

- **Lock-step** is dRRT*'s collision-only execution. dRRT* is not given the pad mutex, so the
  demo prints any pad overlap it has.
- **The asynchronous replay** uses the minimal-interval schedule, which keeps placements apart
  on the pad.
- **Every asynchronous schedule** is checked for collisions every 20 ms, blocks included (carried
  or resting), before it is used. If none passes, the demo falls back to lock-step and marks the
  value with *.
