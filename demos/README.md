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

## Results (demo 4 sweep)

From `python demos/demo4.py --sweep 2,3,4 --seeds 0,1,2 --no_gui`; all 9 (N, seed) runs succeeded,
each in 5-97 s. Makespans are in seconds of velocity-limited motion, given as mean [min, max] over
the 3 seeds. All executions in a run use the same refined paths.

Demo 4 asks whether the minimal intervals avoid whole-action serialization while still
honouring the pad mutex. The baseline is therefore **async full mutex**: whole placements kept
apart on the pad.

| N | async full mutex | async minimal | minimal vs full | pad overlaps (full / minimal) |
|---|---|---|---|---|
| 2 | 6.06 [5.96, 6.20] (0 fb) | 4.63 [4.53, 4.81] (0 fb) | -23.5% | 0 / 0 |
| 3 | 8.60 [7.43, 9.27] (0 fb) | 6.01 [5.65, 6.71] (0 fb) | -28.8% | 0 / 0 |
| 4 | 11.36 [10.66, 11.83] (0 fb) | 7.69 [6.93, 8.61] (0 fb) | -32.4% | 0 / 0 |

The minimal intervals cut the makespan by about a quarter to a third with zero pad overlaps, and
the gain grows with the number of arms.

**Reference: lock-step**, dRRT*'s own execution. It only avoids collisions and is never given the
pad mutex, so it can put two arms on the pad at once.

| N | lock-step | lock-step pad overlaps | async minimal vs lock-step | async motion conflict | refined paths tried |
|---|---|---|---|---|---|
| 2 | 4.68 [4.59, 4.81] | 0 of 3 runs | -1.1% | 4.63 [4.53, 4.81] (0 fb) | 1.0 |
| 3 | 6.49 [6.20, 6.82] | 1 of 3 runs, 1 overlap (1.01 s) | -7.5% | 6.01 [5.65, 6.71] (0 fb) | 1.3 |
| 4 | 7.78 [7.36, 8.33] | 1 of 3 runs, 1 overlap (0.57 s) | -1.4% | 7.76 [6.93, 8.82] (0 fb) | 1.3 |

On average, async minimal is as fast as lock-step or slightly faster. It never violates the pad
mutex, while lock-step violated it in 2 of these 9 runs.

**Planned (Tamer):** minimal is shorter than full mutex at every N: -8.8%, -9.5% and -15.3% for
N = 2, 3, 4. Planned and executed differ because executed durations come from the real refined
paths, which are longer than the representative samples Tamer plans with.

How to read the tables:

- **fb (fallback):** no verified collision-free asynchronous schedule was found, so that run
  executes lock-step and counts with its lock-step makespan and lock-step pad overlaps. No run in
  this sweep fell back.
- **async minimal vs lock-step:** the mean over runs without a minimal fallback.
- **refined paths tried:** each run refines up to 3 paths (`--path_attempts`, a fresh seed each)
  and keeps the first one whose minimal schedule is verified. N = 3 seed 1 and N = 4 seed 2 needed
  a second path.

**5 and 6 arms:** `--sweep` accepts N up to 6, but `python demos/demo4.py --sweep 5,6 --seeds 0
--no_gui` timed out at the 300 s per-run cap for both N = 5 and N = 6 (`--run_timeout` raises the
cap). The sweep reports such runs and leaves them out of the "runs" count. With this
single-shared-pad layout, results are only available up to 4 arms.
