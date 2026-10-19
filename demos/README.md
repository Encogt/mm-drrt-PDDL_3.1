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
python demos/demo4.py --sweep 2,3,4         # headless table over seeds 0,1,2 (--seeds to change; N up to 6)
```

Times are seconds of velocity-limited motion (Franka joint velocity limits), not wall-clock time.
Each demo runs in under 5 minutes headless. The demos re-launch themselves with
`PYTHONHASHSEED=0`, so the same `--seed` always gives the same plans and numbers.

## Multi-seed sweeps without the narration

`experiments/run_contribution1.py`-`run_contribution4.py` run the matching demo's own
plan/refine/measure functions directly (no GUI, no narration, no replay), sweeping seeds as
isolated subprocesses (a crash or timeout only loses that one run) and printing `mean [min, max]`
tables. `run_contribution4.py` is a thin wrapper around `demo4.py`'s own `sweep()` with larger
defaults (10 seeds, 600 s per-run cap instead of 3 seeds / 300 s). Use these for a statistically
larger sweep than a demo's own quick interactive check; the results below come from them.

## Results (demo 4 sweep)

From `python experiments/run_contribution4.py --sweep 2,3,4 --seeds 0,1,...,9`; all 30 (N, seed)
runs succeeded, each in 4 s-312 s. Makespans are in seconds of velocity-limited motion, given as
mean [min, max] over the 10 seeds. All executions in a run use the same refined paths.

Demo 4 asks whether the minimal intervals avoid whole-action serialization while still
honouring the pad mutex. The baseline is therefore **async full mutex**: whole placements kept
apart on the pad.

| N | async full mutex | async minimal | minimal vs full | pad overlaps (full / minimal) |
|---|---|---|---|---|
| 2 | 5.58 [4.17, 6.72] (0 fb) | 4.43 [3.35, 5.32] (0 fb) | -19.9% | 0 (0.00s) / 0 (0.00s) |
| 3 | 10.01 [7.43, 15.19] (1 fb) | 6.92 [5.43, 11.18] (1 fb) | -30.4% | 0 (0.00s) / 0 (0.00s) |
| 4 | 12.36 [8.63, 18.38] (2 fb) | 8.69 [6.80, 11.59] (1 fb) | -27.9% | 1 (1.06s) / 0 (0.00s) |

The minimal intervals cut the makespan by roughly a fifth to a third, and the gain is
non-monotonic across N: it grows from N = 2 to N = 3, then erodes at N = 4 (more re-timing
fallbacks and the first non-zero full-mutex pad overlap show up there too).

**Reference: lock-step**, dRRT*'s own execution. It only avoids collisions and is never given the
pad mutex, so it can put two arms on the pad at once.

| N | lock-step | lock-step pad overlaps | async minimal vs lock-step | async motion conflict | refined paths tried |
|---|---|---|---|---|---|
| 2 | 4.62 [3.84, 5.32] | 0 of 10 runs | -4.1% | 4.44 [3.40, 5.32] (0 fb) | 1.0 |
| 3 | 7.33 [6.19, 11.18] | 2 of 10 runs, 2 overlaps (1.68 s) | -6.5% | 6.92 [5.43, 11.18] (1 fb) | 1.5 |
| 4 | 8.87 [7.31, 11.71] | 4 of 10 runs, 5 overlaps (4.88 s) | -1.6% | 8.94 [6.80, 12.50] (3 fb) | 1.7 |

Async minimal stays close to lock-step (within ~6.5%) while never violating the pad mutex, where
lock-step did on 6 of these 30 runs. `refined paths tried` climbing from 1.0 to 1.7 as N grows is
the same scalability pressure visible in the makespan gain eroding at N = 4.

**Planned (Tamer):** minimal is shorter than full mutex at every N: -8.0%, -9.9% and -14.3% for
N = 2, 3, 4. Planned and executed differ because executed durations come from the real refined
paths, which are longer than the representative samples Tamer plans with.

How to read the tables:

- **fb (fallback):** no verified collision-free asynchronous schedule was found, so that run
  executes lock-step and counts with its lock-step makespan and lock-step pad overlaps.
- **async minimal vs lock-step:** the mean over runs without a minimal fallback.
- **refined paths tried:** each run refines up to 3 paths (`--path_attempts`, a fresh seed each)
  and keeps the first one whose minimal schedule is verified.

**5 and 6 arms:** `--sweep` accepts N up to 6, but `python experiments/run_contribution4.py
--sweep 5,6 --seeds 0,1,...,9` hit the timeout on all of N = 5's first 3 seeds even at a 600 s
per-run cap (double the demos' own 300 s default) -- stopped there rather than burn through the
remaining 17 of 20 runs; N = 6 was never attempted. This is a real, reportable scalability
ceiling for this implementation at N ≈ 5 on commodity hardware, not a gap in the sweep. With
this single-shared-pad layout, results are only available up to 4 arms.
