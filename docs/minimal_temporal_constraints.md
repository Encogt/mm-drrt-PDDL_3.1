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

## Future work: the repair / PDDL-feedback loop

What's implemented here measures a representative trajectory ONCE, offline, before Tamer plans --
it cannot know the exact trajectory the eventual solved plan will execute for a specific object
instance (that's the second approximation above). The natural next step is a closed loop:

1. Plan with the current best `alpha`/`beta` estimate (as today).
2. Execute (or refine via `PlanSkeleton`/dRRT*) and measure the ACTUAL entry/exit times the real,
   plan-specific trajectory produced.
3. If the actual measurement disagrees with the estimate used to plan -- e.g. the real trajectory
   enters the region earlier than assumed, so the mutex window planned for was too narrow -- treat
   that as a **repair** signal: update the region's stored `alpha`/`beta`, and either re-solve the
   affected part of the plan or flag it for re-verification.
4. Feed the corrected value back into the PDDL problem generation for the NEXT planning episode
   (`generate_problem`'s `region_offsets` already accepts exactly this shape -- `sample_region_offsets()`'s
   output -- so wiring a measured-post-hoc value back in is a data-source change, not an interface
   change).

This would close the gap between "representative sample" and "the actual plan's trajectory," and
over repeated runs in a fixed environment, converge the stored offsets toward the true geometry.
It also implies a policy for what "disagreement" should trigger a repair (a fixed tolerance? relative
to the sub-interval's own width?) and how aggressively to re-solve vs. just widen the stored window
defensively -- open design questions, not decided here.
