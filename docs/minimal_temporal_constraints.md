# Minimal Temporal Constraints for Shared-Region Conflicts

When two robots conflict in a shared region `Rk`, the PDDL domain now introduces only the
temporal separation the conflicting intervals actually require:

```
si + beta_ik <= sj + alpha_jk      (or the reverse)
```

instead of forcing the two whole high-level actions apart. This preserves partial overlap between
them whenever the true conflicting sub-intervals allow it. Formally, each action `i` carries a set
of `(region, [entry, exit])` pairs:

```
Oi = { (Rk, [alpha_ik, beta_ik]) }
```

and only the `[alpha_ik, beta_ik]` sub-interval of `i`'s own [start, start+duration] window is
treated as actually occupying `Rk`.

This document describes what's implemented, what's still an approximation, and what's deliberately
left as future work. Code pointers throughout; see each file's own docstrings for the full detail.

## Architecture

PDDL 2.1's durative actions only support `at start` / `at end` / `over all` timing -- there is no
syntax for an arbitrary intra-action timepoint. So `[alpha_ik, beta_ik]` is realized by splitting
`transit`/`transfer` into up to 3 chained sub-actions (`approach` / `occupy` / `depart`), linked by
internal marker fluents, where only the `occupy` phase claims the region's `occupied` mutex.
`mm_drrt/utils/pddl_parser.py` folds a split action's phases back into one logical `transit`/
`transfer` before anything downstream (MM-dRRT's `PlanSkeleton`/dRRT*) ever sees it -- neither
Tamer's solving nor dRRT*'s motion refinement had to change.

`mm_drrt/planner/pddl_domain.py`'s `create_mm_drrt_domain()` supports two ways to supply
`alpha`/`beta`:

- **`use_region_fluents=False`** (scalar mode): `alpha`/`beta` are plain domain-wide constants --
  the same pair for every region, for each of `transit`/`transfer`. Off by default
  (`region_mutex_enabled=False`), reproducing the domain exactly as it was before any of this
  existed (`DurationConflictRaiEnvironment`'s `naive_duration_executor.py` demo depends on that).
- **`use_region_fluents=True`** (fluent mode): `alpha`/`beta` become numeric PDDL fluents
  parameterized by `(robot, fixed-obj)` -- `transit-region-entry-offset(?r ?f)`, etc. -- so a
  different region, or the same region approached by a different robot, can carry a genuinely
  different pair. This is the literal realization of `Oi`'s per-action indexing (two robots
  reaching the same region from different sides can have different geometry). Verified against
  Tamer directly (`unified_planning`'s `set_fixed_duration()` accepts a fluent expression, and
  `Minus(fluentA, fluentB)`, without issue) and correctness-tested end-to-end through
  `pddl_problem_generator.py`.

## Motion-derived offsets

`coordination_regions.py` (`mm_drrt/utils/coordination_regions.py`) builds each region's bounding
volume directly from the live scene: a region's frame position/size (read from `ry.Config`, not
duplicated as constants) extruded upward into an "interaction column" -- the contested airspace
above a surface, not just the surface itself.

`mm_drrt/utils/rai_motion_planner_utils.py`'s `sample_region_offsets()` samples ONE representative
grasp + IK + arm-motion trajectory into/out of a region (reusing the same low-level primitives --
`get_grasp_gen`, `get_placement_gen`, `get_fixed_arm_pick_place_ik_gen`, `get_arm_motion_fn`,
`arm_retrieval_motion` -- the real per-plan path computation already drives, just without that
machinery's plan-specific Action/subgoal bookkeeping, since this has to run BEFORE Tamer has
produced a plan to refine), then measures via forward kinematics the fraction of that trajectory
during which the gripper is actually inside the region's bounding volume
(`trajectory_region_fractions()`). Returns `(alpha, beta)` scaled to the action's declared nominal
duration.

`examples/envs/duration_conflict_rai_env.py`'s `DurationConflictRaiEnvironment.
measure_region_offsets()` orchestrates this for the environment's actual conflict points (both
robots' transit-from-zone and transfer-to-drop_pad), and `main_rai.py --region_offsets_from_motion`
wires it into the CLI, replacing the `--*_region_*_offset` constants with these measured values.

## The current constants are a scoped approximation

Even with motion-derived numbers, what's actually fed to Tamer in the tractable path (see below)
is a **domain-wide scalar pair per action type**, not the fully general per-region fluent. Two
approximations are stacked here, and it's worth being precise about both:

1. **Collapsing multiple regions/robots into one pair.** `measure_region_offsets()` takes the
   union (min entry, max exit) across every `(robot, region)` instance it samples, so no
   individually-measured instance ends up under-protected by the collapse -- but a region with a
   tighter true window than another gets treated as if it were as wide as the worst one measured.
   The fluent mode (above) removes this approximation entirely when it's usable.
2. **A single canonical sample per action type, not the actual plan's trajectory.** Both the
   fluent and scalar modes measure a REPRESENTATIVE grasp/placement/path for a given
   (robot, region, action_type) -- valid, real, IK/motion-planned geometry, but not necessarily
   the exact trajectory the eventual solved plan will execute for a specific object instance. See
   "Future work" below for what closing this gap would take.

Before this work, the offsets were hand-picked CLI constants (`--transit_region_entry_offset 3`,
etc.) with no connection to real geometry at all. Both approximations above are strictly tighter
than that starting point -- but neither is "the paper's `Oi`" in full generality yet.

## A practical constraint: Tamer's search does not scale to full-generality at this environment's size

Empirically (not from the papers/docs -- discovered by running it): Tamer's search hangs well past
150 seconds on `DurationConflictRaiEnvironment`'s full size (2 robots x 2 objects x 3 surfaces) once
BOTH `transit` and `transfer` are fully split into all 3 phases (approach+occupy+depart) --
reproduced with both the fluent mode AND plain float constants, so this is a grounding/search-space
cost from having 6 action templates at this environment's branching factor, not something specific
to numeric fluents.

What DOES solve in well under a second at this same scale: narrowing only ONE side of each action
type's window (transit: `entry_offset` only, `exit_offset` stays at the full duration; transfer:
`exit_offset` only, `entry_offset` stays at 0) -- 2 phases per action type, 4 templates total. This
is what `compare_region_constraints_rai_env.py` and `main_rai.py --region_offsets_from_motion`
actually run with: the real measured values, on the one side of each action type that's affordable
at this scale, conservative (never narrowed) on the other side.

This is a genuine, currently-unresolved scaling limit of the specific solving path (Tamer's search
over this domain shape at this environment's grounding size), not a soundness issue in the domain
or the offset-measurement mechanism -- both are verified correct at smaller scale (single robot /
fewer surfaces) and via direct plan validation. Reducing to one side per action type is itself
listed above as part of the "scoped approximation."

`main_rai.py` now tries all four offsets first anyway (`--region_offset_mode four`, the default) and
falls back to the two-offset reduction when Tamer exceeds `--pddl_timeout`. Tamer itself ignores
`timeout` (`UserWarning: Tamer does not support timeout`), so `TamerPDDLPlanner` runs the solve in a
forked child process and kills it on expiry (`_solve_with_wall_clock_timeout`). On this environment
the four-offset problem still times out (confirmed at 15s, 20s and 300s), in both the fraction and
motion-time units, so the fallback is what actually runs here; `--region_offset_mode two` skips the
wasted attempt.

## Validation

- `python compare_region_constraints.py` -- isolated demo problem
  (`region_overlap_demo_problem.pddl`), asserts 5 invariants (both domains solve, old full-mutex
  plan has zero cross-robot overlap, new plan has genuine overlap, new makespan is strictly
  shorter).
- `python compare_region_constraints_rai_env.py` -- the real `DurationConflictRaiEnvironment`,
  three configurations (stock / full mutex / motion-derived minimal constraint), asserting the
  motion-derived run's makespan lands strictly between the stock and full-mutex baselines. Uses
  the one-narrowed-side-per-action-type reduction (see above).
- `python compare_region_fluents_demo.py` -- the dedicated
  `examples/envs/region_coordination_demo_rai_env.py` example, exercising the FULLY GENERAL
  per-(robot, region) fluent mode (both sides narrowed) with real, per-robot-DISTINCT measured
  data -- something the scalar mode/reduction above can't express (it collapses to one shared
  pair). Asserts the two robots' measured windows are actually distinct and that their solved
  occupy phases don't overlap. Runs at the Tamer level only (a transfer-only problem, so it can't
  go through `PlanSkeleton`/dRRT*'s strict pick-then-place assumption) -- see that script's
  docstring for the full reasoning.
- `python main_rai.py --env_type exp_region_coordination_demo --num_robots 2 --num_objs 2
  --use_pddl_planner --region_mutex_enabled --region_offsets_from_motion` -- the full, untouched
  pipeline (Tamer -> `PlanSkeleton`/dRRT* -> composite path) on the dedicated example environment,
  confirming the real motion-refinement black box accepts and executes a plan built from
  motion-derived offsets. (`--env_type exp_duration_conflict_rai` also still works identically --
  the dedicated environment is a thin, same-geometry subclass; see that module's docstring.)

## The repair loop (implemented)

The gap described above -- a representative trajectory measured ONCE, offline, before Tamer
plans, vs. the actual trajectory the eventual solved plan executes for a specific object instance
-- is now closed by a real feedback loop, not just sketched:

1. **Plan** with the current best `alpha`/`beta` estimate, same as before
   (`measure_region_offsets()`), now merged with whatever a PRIOR run already learned (step 4).
2. **Measure the real, plan-specific trajectory.** After `PlanSkeleton.plan_refinement()` produces
   a `composite_path`, `DurationConflictRaiEnvironment.measure_executed_region_offsets()` extracts
   each robot's actual dRRT*-refined joint-space path (concatenating `node.sub_local_paths`),
   splits it into its transit/transfer segments (the grasp event, then the robot's exact return to
   `carry_conf` -- confirmed via instrumentation to be an exact, zero-distance match, not an
   approximate threshold), and runs `trajectory_region_fractions()` on each segment -- the SAME
   measurement primitive `sample_region_offsets()` uses at planning time, just against the real
   executed path instead of a representative sample.
3. **Repair.** `mm_drrt/utils/temporal_repair.py`'s `repair_region_offsets()` compares planned vs.
   measured with a fixed, absolute tolerance (0.5s, not scaled to the sub-interval's own width --
   deliberately simple, and avoids a tight sub-interval's sensitivity scaling down to noise level).
   A measured offset outside tolerance in the dangerous direction (entered earlier / exited later
   than planned) is flagged `violated`; either way, the stored `(alpha, beta)` widens to the union
   of planned and measured -- unconditionally, so the stored window only ever widens from a
   real-world observation, never narrows.

   **Decision on what a violation does:** it never fails or aborts the run. dRRT*'s own
   inter-robot collision checking already verified THIS run's composite path is collision-free
   regardless of what the PDDL-level mutex window assumed -- a violation means the SCHEDULE was built
   from offsets that under-covered the real geometry. What happens next is `--repair_strategy`
   (see "Repairing within the run" below). A violation is judged against the window Tamer actually
   enforced (under the two-offset reduction the transit window runs to the end of the action), but
   what is stored is the union of the measured planned offsets and the executed ones, so a
   two-offset run never caches a full-duration window as if it had been measured.
4. **Feed forward.** `mm_drrt/utils/region_offsets_cache.py` persists the widened `(alpha, beta)`
   to a small JSON file keyed by environment class name. The NEXT run loads it and merges it with
   a fresh measurement before planning (step 1) -- confirmed end-to-end over two consecutive runs:
   run 1 measured, executed, widened, and saved; run 2 loaded that cache, merged it in, planned
   from the merged value, and widened again from there. With `--durations_from_motion` entries are
   `[alpha, beta, duration]` in seconds under a separate `<EnvClass>:motion_time` key.

Run via `main_rai.py --region_offsets_from_motion` (same flag as before -- the repair step runs
automatically whenever it's set and the environment implements `measure_executed_region_offsets()`).

## Least-commitment check

After every Tamer solve (with `--region_mutex_enabled`), `schedule_repair.check_least_commitment()`
takes the solved schedule (`parse_pddl_plan(..., return_schedule=True)`, which now keeps each
action's occupy-phase interval) and, for every pair of different robots' actions on the same region,
compares the actual start gap with `derive_minimal_constraint()`'s theoretical minimum. Each pair is
`tight` (slack within 0.05s -- Tamer separates mutex-ordered happenings by 0.01), `not-binding`, or
has its extra delay explained by `robot-sequence` / `object-handoff` / `time-zero`; anything else is
`unexplained`, i.e. NOT least-commitment. It also flags pairs where Tamer chose the more expensive
order. On `DurationConflictRaiEnvironment` every run so far reports the drop_pad pair as tight in the
minimal order (e.g. gap 6.112 vs. minimum 6.102; 0.818 vs. 0.808 in motion time).

## Repairing within the run (Steps 5a/5b)

`--repair_strategy` (default `retime`) decides what a violation does:

- **`retime` (Step 5a).** `schedule_repair.retime_schedule()` keeps the symbolic plan -- same
  actions, same per-robot, per-object and per-region order -- and re-times it under the measured
  offsets/durations. Every constraint is a difference constraint `s_v - s_u >= w`: same-robot
  sequencing and cross-robot handoffs (`s_b - s_a >= d_a`), and the region constraint
  `s_j - s_i >= beta_ik - alpha_jk` in Tamer's order. The earliest start times are a longest-path
  solve from a time-zero source (Bellman-Ford); a positive cycle -- or exceeding
  `--retime_max_makespan_factor` x the solved makespan, if set -- is infeasible and falls through to
  `resolve`. Example (fraction units): the measured transfer window [3.90, 6.19] instead of the
  planned [0, 6.10] moves the second transfer from 16.13 to 12.30 and the makespan 26.14 -> 22.31.
- **`resolve` (Step 5b).** Restore the world, widen the offsets (and durations), regenerate the
  problem, re-run Tamer and refinement, up to `--max_resolve_attempts` (default 2).
- **`report`.** The previous behaviour: widen and cache only.

`--repair_tolerance` (default 0.5s) sets what counts as a violation.

## Motion-derived durations (C1)

`--durations_from_motion` times every trajectory under velocity-limited execution
(`mm_drrt/utils/motion_timing.py`: each step takes `max_j |dq_j| / vmax_j`, Franka limits x
`--joint_velocity_scale`) instead of using waypoint-index fractions of a declared duration. Index
fractions depend on interpolation density -- inside one dRRT* composite node the two arms carry
different waypoint counts (30 vs. 19 observed) -- which is consistent with the +0.5s transfer
`entry_diff` the fraction mode reported. The measured durations become the PDDL `:duration`s;
`alpha`/`beta` are seconds into the action.

On the executed side, `executed_action_timeline()` splits each robot's composite path at its
`carry_conf` visits (one segment per non-return action) and synchronises robots at composite nodes.
Result on `DurationConflictRaiEnvironment`: the representative pre-planning samples (~1.33s) underestimate
the executed dRRT* trajectories (~2.76s transit, ~2.90s transfer) by about 2x -- a real violation. One
`resolve` with the widened durations converges: re-planned 2.764 / 2.895s vs. re-executed 2.743 / 2.891s.

## Motion-based conflict detection (C2)

`--detect_motion_conflicts` sweeps every pair of different robots' executed actions, waypoint against
waypoint (`--conflict_samples 0` = every waypoint; uniform subsampling can skip a short colliding
window), through dRRT*'s own `get_inter_robots_collision_fn`. This asks whether ANY time alignment of
the two motions could collide -- what a mutex has to rule out. For a colliding pair, the conflict
interval in each action (widened to the neighbouring samples) replaces the region-volume window.
The report lists shared-region pairs that never collide ("mutex unnecessary") and colliding pairs
with no shared region ("unanticipated"); `retime` then constrains only the colliding pairs, over
their conflict intervals. Whether the drop_pad pair collides varies with the refined paths: one run
found 276 colliding waypoint pairs (a4 [0.40, 1.01] x a5 [0.74, 1.77]), others found the grippers
within ~5cm but no arm-arm contact. Approximation: attached blocks are not moved with the gripper
during the sweep.

## Demos and the N-arm scene

`demos/` holds one GUI walkthrough per contribution (see `demos/README.md`); the pipeline they drive
is `mm_drrt/pipeline_rai.py`, which `main_rai.py` now also imports. Contribution 4 needed more than
two arms, so `examples/envs/round_table_rai_env.py` (`--env_type exp_round_table`) generates a scene
with N Frankas around one shared pad. Making N > 2 work surfaced four issues, fixed as follows:

- **Zones next to neighbours.** pandasTable.g's zone offset puts a zone ~0.25 m from the next arm's
  base at N = 4; zones now sit on the outer side of each arm.
- **Grasp carry pose.** `get_grasp_gen` built every grasp with the module-default carry pose; it now
  uses the robot's own `carry_conf`. (Turning the round-table arms' carry pose aside was tried and
  reverted: sampled pick/place paths then hit an idle neighbour on 5-10% of their waypoints, against
  none with the standard pose.)
- **dRRT* parent lookup.** `get_parent_node_index` returns the first tree node with a matching
  configuration in any subproblem. Every fixed-arm action starts and ends at `carry_conf`, so when
  all arms are back at carry together, the root matches, and the retraced composite path silently
  skipped those actions (seen with 3 arms: a "solved" path in which two arms never moved).
  `rai_drrt_star._parent_node_index` takes the most recent match in the current subproblem.
- **Global order-constraint check.** `is_violate_order_constraints` disables `connect_to_target`
  for every robot while any one waits on a precedence, which stalls the search with 4 arms.
  `--drrt_order_constraints handoff` passes dRRT* only same-object handoffs; region timing is then
  handled by the schedule (re-timing and the verified asynchronous execution).

dRRT* still has no backtracking: with several arms it can greedily move every arm onto the pad at
once and then find no collision-free retreat. demo 4 bounds each attempt (`--drrt_time_limit`, 60 s
by default) and retries with a new seed.

**Asynchronous execution** (`motion_timing.async_schedule`). This re-times Tamer's actions with
their measured durations, then sweeps the result on a 20 ms grid for robot-robot collisions. A
collision between two moving actions adds that pair's motion-conflict window. A collision with an
idle robot (waiting at `carry_conf`) is resolved differently, using the lock-step dRRT* path as a
witness: the moving action's colliding part is constrained to fall inside the stretch of one of the
idle robot's own actions during which that robot is clear of it. These are still difference
constraints, so the re-timing remains a shortest-path solve. The resulting makespans are in
`demos/README.md`. Against full serialization the minimal intervals always win. Against lock-step
they win at N = 3 but not at N = 4.

## Future work

- **Downstream enforcement.** Sub-interval precedence exists in Tamer's schedule and in the re-timed
  schedule, but PlanSkeleton/dRRT* still only receive whole-action precedence
  (`init_order_constraints`), and dRRT* has no notion of time. Enforcing a partial overlap during
  execution needs either timed sub-interval constraints in dRRT*'s composite search or a timed
  executor that follows the re-timed schedule.
- **Per-(robot, object, region) timing.** Durations and offsets are still collapsed per action type
  (min entry, max exit, max duration). The fluent mode already has per-(robot, region) offsets;
  durations would need the same treatment in `pddl_domain.py`.
- **PDDL-level mutex gating from C2.** Conflict detection feeds re-timing only; the next Tamer solve
  still applies `occupied` to every shared region. Dropping it for regions with no colliding pair
  needs a per-region gate in the domain (e.g. an occupy variant that skips `occupied`) and parser
  support for it.
- **Unanticipated conflicts back into PDDL.** A colliding pair with no shared region is reported and
  constrained in re-timing, but not compiled into a new PDDL constraint.
- **Four offsets at this scale.** See the scaling section: the four-offset problem still times out
  here, so the two-offset fallback is what runs.
