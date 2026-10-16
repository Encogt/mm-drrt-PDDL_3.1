"""
PDDL 2.1 Temporal Domain Definition for MM-dRRT Manipulation Tasks

Uses durative actions (a PDDL 2.1 feature) so the planner can reason about
temporal overlap between robots. Object location is a boolean predicate
(obj-location m f), not a PDDL 3.1 object fluent -- kept consistent with the
Fast Downward path (mm_drrt/planner/classical_pddl.py), which already
represents it the same way (translated to its own `at` predicate).
"""

import os
from unified_planning.shortcuts import *


def create_mm_drrt_domain(transit_duration=10, transfer_duration=10, region_mutex_enabled=False,
                          transit_region_entry_offset=0, transit_region_exit_offset=None,
                          transfer_region_entry_offset=0, transfer_region_exit_offset=None,
                          use_region_fluents=False):
    """
    Create MM-dRRT manipulation domain using UPF's domain builder.

    Uses PDDL 2.1 durative actions so the planner can schedule multiple
    robots in parallel based on action durations.

    region_mutex_enabled=False (the default) reproduces the domain EXACTLY as it was before
    region-conflict modeling existed: no `occupied` fluent at all, so two actions on the same
    fixed-obj (even from different robots/objects) are never forced apart by the PDDL domain
    itself -- see mm_drrt.examples.envs.duration_conflict_rai_env.DurationConflictRaiEnvironment,
    whose whole demo (a wrong --transfer_duration causing an undetected collision) depends on the
    "stock" domain NOT protecting a shared surface. Passing a non-default region offset while
    region_mutex_enabled is False raises, rather than silently ignoring it.

    With region_mutex_enabled=True, `occupied ?f` is claimed for the sub-interval
    [start + entry_offset, start + exit_offset] during which the action ACTUALLY occupies the
    shared region -- i.e. alpha_ik/beta_ik in the paper's Oi = {(Rk, [alpha_ik, beta_ik])},
    "si + beta_ik <= sj + alpha_jk, or the reverse" -- instead of the action's full duration. Two
    robots conflicting in the same region only get forced apart across those sub-intervals, so the
    rest of their high-level actions (e.g. one robot's approach overlapping another's retreat) can
    still run in parallel.

    Two ways to get alpha_ik/beta_ik into this domain, and PDDL 2.1's at-start/at-end/over-all
    timing model forces BOTH to realize them the same way: splitting transit/transfer into up to 3
    chained sub-actions (approach / occupy / depart) linked by internal marker fluents (there's no
    PDDL 2.1 syntax for an arbitrary intra-action timepoint). mm_drrt/utils/pddl_parser.py folds a
    split action's phases back into one before anything downstream sees them either way.

    - use_region_fluents=False (default): transit_region_entry_offset/transit_region_exit_offset
      (and the transfer_ equivalents) are plain GLOBAL constants, the same for every fixed-obj --
      a single hand-declared (alpha, beta) pair per action type. This is the scoped approximation:
      real regions differ in size/reach geometry, but this mode can't express that. Defaults (0,
      duration) reproduce a full-duration mutex exactly, and the domain collapses back to a single
      'transit'/'transfer' action with no split at all -- both offsets must move away from their
      defaults for partial overlap to become possible.
    - use_region_fluents=True: alpha_ik/beta_ik become PER-(ROBOT, FIXED-OBJ) numeric fluents
      (transit-region-entry-offset(?r ?f), etc.) instead of domain-wide constants -- matching
      Oi's indexing by action i (a specific robot doing a specific move), since two robots can
      approach the very same region from different sides with genuinely different geometry -- see
      mm_drrt/utils/rai_motion_planner_utils.py's sample_region_offsets() for how those per-object
      values get measured from a real IK/motion-planned trajectory, and
      mm_drrt/planner/pddl_problem_generator.py for where they get set as this fluent's per-object
      initial value. transit_region_entry_offset/transit_region_exit_offset (and the transfer_
      equivalents) become each fluent's DEFAULT value instead -- the fallback for any fixed-obj no
      motion sample was measured for. The domain always splits into all 3 phases in this mode
      (a specific object's measured offsets can still be degenerate -- e.g. entry_offset=0 -- which
      Tamer accepts fine as a zero-duration phase for that grounded instance; it just can't be
      decided statically per-domain any more since different objects can disagree).

    Returns:
        dict with keys: boolean_fluents, numeric_fluents, actions, types
        numeric_fluents is a list of (fluent, default_value) pairs (empty unless
        use_region_fluents=True) -- register each with
        problem.add_fluent(fluent, default_initial_value=default_value).
    """
    if transit_region_exit_offset is None:
        transit_region_exit_offset = transit_duration
    if transfer_region_exit_offset is None:
        transfer_region_exit_offset = transfer_duration

    if not region_mutex_enabled:
        if use_region_fluents:
            raise ValueError("use_region_fluents=True requires region_mutex_enabled=True.")
        if (transit_region_entry_offset, transit_region_exit_offset) != (0, transit_duration) or \
           (transfer_region_entry_offset, transfer_region_exit_offset) != (0, transfer_duration):
            raise ValueError(
                "region offsets were set but region_mutex_enabled=False -- they would be "
                "silently ignored. Pass region_mutex_enabled=True to actually enforce them.")

    for label, entry, exit_, duration in (
        ('transit', transit_region_entry_offset, transit_region_exit_offset, transit_duration),
        ('transfer', transfer_region_entry_offset, transfer_region_exit_offset, transfer_duration),
    ):
        if not (0 <= entry <= exit_ <= duration):
            raise ValueError(
                f"{label} region offsets must satisfy 0 <= entry_offset <= exit_offset <= "
                f"duration, got entry={entry}, exit={exit_}, duration={duration}")

    Robot      = UserType('robot')
    MovableObj = UserType('movable-obj')
    FixedObj   = UserType('fixed-obj')

    # Boolean predicates
    robot_at_base      = Fluent('robot-at-base',       BoolType(), r=Robot)
    robot_free         = Fluent('robot-free',           BoolType(), r=Robot)
    holding            = Fluent('holding',              BoolType(), r=Robot, m=MovableObj)
    obj_clear          = Fluent('obj-clear',            BoolType(), m=MovableObj)
    surface_accessible = Fluent('surface-accessible',   BoolType(), f=FixedObj)
    robot_can_reach    = Fluent('robot-can-reach',      BoolType(), r=Robot, f=FixedObj)

    # Object location as a boolean predicate: true while m rests on f.
    obj_location = Fluent('obj-location', BoolType(), m=MovableObj, f=FixedObj)

    # True only while a robot is actually occupying a shared region right now (see docstring).
    occupied = Fluent('occupied', BoolType(), f=FixedObj)

    # Chain markers linking a split action's phases for one (robot, obj, surface) instance.
    # Purely internal bookkeeping; only created/used when a split is actually needed.
    transit_entered         = Fluent('transit-entered',         BoolType(), r=Robot, m=MovableObj, f=FixedObj)
    transit_occupied_done   = Fluent('transit-occupied-done',   BoolType(), r=Robot, m=MovableObj, f=FixedObj)
    transfer_entered        = Fluent('transfer-entered',        BoolType(), r=Robot, m=MovableObj, f=FixedObj)
    transfer_occupied_done  = Fluent('transfer-occupied-done',  BoolType(), r=Robot, m=MovableObj, f=FixedObj)

    # Per-fixed-obj alpha_ik/beta_ik -- only built/registered when use_region_fluents=True.
    transit_entry_offset_f  = Fluent('transit-region-entry-offset',  RealType(), r=Robot, f=FixedObj)
    transit_exit_offset_f   = Fluent('transit-region-exit-offset',   RealType(), r=Robot, f=FixedObj)
    transfer_entry_offset_f = Fluent('transfer-region-entry-offset', RealType(), r=Robot, f=FixedObj)
    transfer_exit_offset_f  = Fluent('transfer-region-exit-offset',  RealType(), r=Robot, f=FixedObj)

    boolean_fluents = [robot_at_base, robot_free, holding, obj_clear,
                       surface_accessible, robot_can_reach, obj_location]
    numeric_fluents = []
    if region_mutex_enabled:
        boolean_fluents.append(occupied)
    if use_region_fluents:
        numeric_fluents = [
            (transit_entry_offset_f, float(transit_region_entry_offset)),
            (transit_exit_offset_f, float(transit_region_exit_offset)),
            (transfer_entry_offset_f, float(transfer_region_entry_offset)),
            (transfer_exit_offset_f, float(transfer_region_exit_offset)),
        ]

    actions = []

    # ---- transit(r, m, from): pick object m from surface from -------------------------------
    transit_has_approach = use_region_fluents or (transit_region_entry_offset > 0)
    transit_has_depart   = use_region_fluents or (transit_region_exit_offset < transit_duration)
    transit_occupy_duration = transit_region_exit_offset - transit_region_entry_offset

    if not transit_has_approach and not transit_has_depart:
        transit = DurativeAction('transit', r=Robot, m=MovableObj, from_f=FixedObj)
        transit.set_fixed_duration(transit_duration)
        r, m, from_f = transit.parameter('r'), transit.parameter('m'), transit.parameter('from_f')
        transit.add_condition(StartTiming(), robot_free(r))
        transit.add_condition(StartTiming(), obj_location(m, from_f))
        transit.add_condition(StartTiming(), obj_clear(m))
        transit.add_condition(ClosedTimeInterval(StartTiming(), EndTiming()), surface_accessible(from_f))
        transit.add_condition(ClosedTimeInterval(StartTiming(), EndTiming()), robot_can_reach(r, from_f))
        if region_mutex_enabled:
            transit.add_condition(StartTiming(), Not(occupied(from_f)))
        transit.add_effect(StartTiming(), robot_free(r), False)
        transit.add_effect(StartTiming(), obj_clear(m),  False)
        transit.add_effect(StartTiming(), obj_location(m, from_f), False)
        if region_mutex_enabled:
            transit.add_effect(StartTiming(), occupied(from_f), True)
            transit.add_effect(EndTiming(),   occupied(from_f), False)
        transit.add_effect(EndTiming(),   holding(r, m), True)
        actions.append(transit)
    else:
        boolean_fluents.append(transit_entered)
        if transit_has_depart:
            boolean_fluents.append(transit_occupied_done)

        if transit_has_approach:
            approach = DurativeAction('transit-approach', r=Robot, m=MovableObj, from_f=FixedObj)
            r, m, from_f = approach.parameter('r'), approach.parameter('m'), approach.parameter('from_f')
            approach.set_fixed_duration(transit_entry_offset_f(r, from_f) if use_region_fluents
                                       else transit_region_entry_offset)
            approach.add_condition(StartTiming(), robot_free(r))
            approach.add_condition(StartTiming(), obj_location(m, from_f))
            approach.add_condition(StartTiming(), obj_clear(m))
            approach.add_condition(ClosedTimeInterval(StartTiming(), EndTiming()), surface_accessible(from_f))
            approach.add_condition(ClosedTimeInterval(StartTiming(), EndTiming()), robot_can_reach(r, from_f))
            approach.add_effect(StartTiming(), robot_free(r), False)
            approach.add_effect(StartTiming(), obj_clear(m),  False)
            approach.add_effect(StartTiming(), obj_location(m, from_f), False)
            approach.add_effect(EndTiming(), transit_entered(r, m, from_f), True)
            actions.append(approach)

        occupy = DurativeAction('transit-occupy', r=Robot, m=MovableObj, from_f=FixedObj)
        r, m, from_f = occupy.parameter('r'), occupy.parameter('m'), occupy.parameter('from_f')
        occupy.set_fixed_duration(Minus(transit_exit_offset_f(r, from_f), transit_entry_offset_f(r, from_f))
                                 if use_region_fluents else transit_occupy_duration)
        if transit_has_approach:
            occupy.add_condition(StartTiming(), transit_entered(r, m, from_f))
            occupy.add_effect(StartTiming(), transit_entered(r, m, from_f), False)
        else:
            occupy.add_condition(StartTiming(), robot_free(r))
            occupy.add_condition(StartTiming(), obj_location(m, from_f))
            occupy.add_condition(StartTiming(), obj_clear(m))
            occupy.add_effect(StartTiming(), robot_free(r), False)
            occupy.add_effect(StartTiming(), obj_clear(m),  False)
            occupy.add_effect(StartTiming(), obj_location(m, from_f), False)
        occupy.add_condition(ClosedTimeInterval(StartTiming(), EndTiming()), surface_accessible(from_f))
        occupy.add_condition(ClosedTimeInterval(StartTiming(), EndTiming()), robot_can_reach(r, from_f))
        occupy.add_condition(StartTiming(), Not(occupied(from_f)))
        occupy.add_effect(StartTiming(), occupied(from_f), True)
        occupy.add_effect(EndTiming(),   occupied(from_f), False)
        if transit_has_depart:
            occupy.add_effect(EndTiming(), transit_occupied_done(r, m, from_f), True)
        else:
            occupy.add_effect(EndTiming(), holding(r, m), True)
        actions.append(occupy)

        if transit_has_depart:
            depart = DurativeAction('transit-depart', r=Robot, m=MovableObj, from_f=FixedObj)
            r, m, from_f = depart.parameter('r'), depart.parameter('m'), depart.parameter('from_f')
            depart.set_fixed_duration(Minus(transit_duration, transit_exit_offset_f(r, from_f))
                                     if use_region_fluents else transit_duration - transit_region_exit_offset)
            depart.add_condition(StartTiming(), transit_occupied_done(r, m, from_f))
            depart.add_condition(ClosedTimeInterval(StartTiming(), EndTiming()), surface_accessible(from_f))
            depart.add_condition(ClosedTimeInterval(StartTiming(), EndTiming()), robot_can_reach(r, from_f))
            depart.add_effect(StartTiming(), transit_occupied_done(r, m, from_f), False)
            depart.add_effect(EndTiming(), holding(r, m), True)
            actions.append(depart)

    # ---- transfer(r, m, to): place object m on surface to -----------------------------------
    transfer_has_approach = use_region_fluents or (transfer_region_entry_offset > 0)
    transfer_has_depart   = use_region_fluents or (transfer_region_exit_offset < transfer_duration)
    transfer_occupy_duration = transfer_region_exit_offset - transfer_region_entry_offset

    if not transfer_has_approach and not transfer_has_depart:
        transfer = DurativeAction('transfer', r=Robot, m=MovableObj, to_f=FixedObj)
        transfer.set_fixed_duration(transfer_duration)
        r, m, to_f = transfer.parameter('r'), transfer.parameter('m'), transfer.parameter('to_f')
        transfer.add_condition(StartTiming(), holding(r, m))
        transfer.add_condition(ClosedTimeInterval(StartTiming(), EndTiming()), surface_accessible(to_f))
        transfer.add_condition(ClosedTimeInterval(StartTiming(), EndTiming()), robot_can_reach(r, to_f))
        if region_mutex_enabled:
            transfer.add_condition(StartTiming(), Not(occupied(to_f)))
            transfer.add_effect(StartTiming(), occupied(to_f), True)
            transfer.add_effect(EndTiming(), occupied(to_f), False)
        transfer.add_effect(EndTiming(), robot_free(r),    True)
        transfer.add_effect(EndTiming(), holding(r, m),    False)
        transfer.add_effect(EndTiming(), obj_clear(m),     True)
        transfer.add_effect(EndTiming(), obj_location(m, to_f), True)
        actions.append(transfer)
    else:
        if transfer_has_approach:
            boolean_fluents.append(transfer_entered)
        if transfer_has_depart:
            boolean_fluents.append(transfer_occupied_done)

        if transfer_has_approach:
            approach = DurativeAction('transfer-approach', r=Robot, m=MovableObj, to_f=FixedObj)
            r, m, to_f = approach.parameter('r'), approach.parameter('m'), approach.parameter('to_f')
            approach.set_fixed_duration(transfer_entry_offset_f(r, to_f) if use_region_fluents
                                       else transfer_region_entry_offset)
            approach.add_condition(StartTiming(), holding(r, m))
            approach.add_condition(ClosedTimeInterval(StartTiming(), EndTiming()), surface_accessible(to_f))
            approach.add_condition(ClosedTimeInterval(StartTiming(), EndTiming()), robot_can_reach(r, to_f))
            approach.add_effect(EndTiming(), transfer_entered(r, m, to_f), True)
            actions.append(approach)

        occupy = DurativeAction('transfer-occupy', r=Robot, m=MovableObj, to_f=FixedObj)
        r, m, to_f = occupy.parameter('r'), occupy.parameter('m'), occupy.parameter('to_f')
        occupy.set_fixed_duration(Minus(transfer_exit_offset_f(r, to_f), transfer_entry_offset_f(r, to_f))
                                 if use_region_fluents else transfer_occupy_duration)
        if transfer_has_approach:
            occupy.add_condition(StartTiming(), transfer_entered(r, m, to_f))
            occupy.add_effect(StartTiming(), transfer_entered(r, m, to_f), False)
        else:
            occupy.add_condition(StartTiming(), holding(r, m))
        occupy.add_condition(ClosedTimeInterval(StartTiming(), EndTiming()), surface_accessible(to_f))
        occupy.add_condition(ClosedTimeInterval(StartTiming(), EndTiming()), robot_can_reach(r, to_f))
        occupy.add_condition(StartTiming(), Not(occupied(to_f)))
        occupy.add_effect(StartTiming(), occupied(to_f), True)
        occupy.add_effect(EndTiming(),   occupied(to_f), False)
        if transfer_has_depart:
            occupy.add_effect(EndTiming(), transfer_occupied_done(r, m, to_f), True)
        else:
            occupy.add_effect(EndTiming(), robot_free(r), True)
            occupy.add_effect(EndTiming(), holding(r, m), False)
            occupy.add_effect(EndTiming(), obj_clear(m), True)
            occupy.add_effect(EndTiming(), obj_location(m, to_f), True)
        actions.append(occupy)

        if transfer_has_depart:
            depart = DurativeAction('transfer-depart', r=Robot, m=MovableObj, to_f=FixedObj)
            r, m, to_f = depart.parameter('r'), depart.parameter('m'), depart.parameter('to_f')
            depart.set_fixed_duration(Minus(transfer_duration, transfer_exit_offset_f(r, to_f))
                                     if use_region_fluents else transfer_duration - transfer_region_exit_offset)
            depart.add_condition(StartTiming(), transfer_occupied_done(r, m, to_f))
            depart.add_condition(ClosedTimeInterval(StartTiming(), EndTiming()), surface_accessible(to_f))
            depart.add_condition(ClosedTimeInterval(StartTiming(), EndTiming()), robot_can_reach(r, to_f))
            depart.add_effect(StartTiming(), transfer_occupied_done(r, m, to_f), False)
            depart.add_effect(EndTiming(), robot_free(r), True)
            depart.add_effect(EndTiming(), holding(r, m), False)
            depart.add_effect(EndTiming(), obj_clear(m), True)
            depart.add_effect(EndTiming(), obj_location(m, to_f), True)
            actions.append(depart)

    return {
        'boolean_fluents': boolean_fluents,
        'numeric_fluents': numeric_fluents,
        'actions': actions,
        'types': {'robot': Robot, 'movable-obj': MovableObj, 'fixed-obj': FixedObj},
        'region_offset_fluents': {
            'transit_entry': transit_entry_offset_f, 'transit_exit': transit_exit_offset_f,
            'transfer_entry': transfer_entry_offset_f, 'transfer_exit': transfer_exit_offset_f,
        },
    }


def get_domain_file_path():
    domain_file = os.path.join(
        os.path.dirname(__file__),
        '../pddl/domains/mm_drrt_manipulation.pddl'
    )
    return os.path.abspath(domain_file)
