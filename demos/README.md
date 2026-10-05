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
Each demo runs in under 5 minutes headless. The demos re-launch themselves with
`PYTHONHASHSEED=0`, so the same `--seed` always gives the same plans and numbers.

## Results (demo 4 sweep)

From `python demos/demo4.py --sweep 2,3,4 --seeds 0,1,2`. Makespans are in seconds of
velocity-limited motion, given as mean [min, max] over 3 seeds; each (N, seed) run took 5-80 s.

**Planned (Tamer): full mutex vs minimal intervals.** Minimal is shorter at every N and every
seed.

| N | full mutex | minimal | minimal vs full |
|---|---|---|---|
| 2 | 4.02 [3.93, 4.06] | 3.66 [3.60, 3.69] | -8.8% |
| 3 | 7.83 [6.60, 8.64] | 7.09 [5.87, 7.88] | -9.5% |
| 4 | 7.09 [6.50, 7.85] | 6.00 [5.47, 6.69] | -15.3% |

**Executed: the same refined paths, lock-step vs asynchronous.** "fb" counts runs where no
verified collision-free asynchronous schedule was found, so lock-step was used instead; those runs
count with their lock-step makespan. The last column is the mean over runs without a fallback.

| N | lock-step | async, full mutex | async, minimal | async, motion conflict | async minimal vs lock-step |
|---|---|---|---|---|---|
| 2 | 4.68 [4.59, 4.81] | 6.06 [5.96, 6.20] (0 fb) | 4.63 [4.53, 4.81] (0 fb) | 4.63 [4.53, 4.81] (0 fb) | -1.1% |
| 3 | 6.26 [6.12, 6.45] | 9.06 [8.79, 9.27] (0 fb) | 5.82 [5.65, 6.12] (1 fb) | 5.93 [5.65, 6.46] (0 fb) | -10.4% |
| 4 | 7.39 [7.15, 7.66] | 11.35 [10.66, 11.81] (0 fb) | 7.20 [6.93, 7.53] (1 fb) | 7.20 [6.93, 7.53] (1 fb) | -3.8% |

How to read this:

- The schedule-level gain of the minimal intervals over a whole-action mutex is consistent, and
  grows with N.
- On the executed paths, asynchronous minimal always beats asynchronous full serialization.
- Against lock-step, asynchronous minimal helps clearly at N = 3 (-10.4%), modestly at N = 4
  (-3.8%) and barely at N = 2 (-1.1%).
- It still falls back to lock-step in 1 of 3 seeds at both N = 3 and N = 4. At N = 4 (seed 2)
  the reason is geometric: one arm's resting pose lies in another arm's placement path, and no
  re-timing of the same paths avoids it. Lock-step avoided it only because that arm happened to
  be busy at that moment. Lock-step here is dRRT*'s collision-only execution: it is not given the pad
  mutex, and it prints any pad overlap it has.
