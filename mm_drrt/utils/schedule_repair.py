"""C3 downstream of Tamer: checking that a solved schedule is least-commitment, and repairing it
(Step 5a) by RE-TIMING the existing actions instead of re-planning.

Both work on the solved schedule pddl_parser.parse_pddl_plan(..., return_schedule=True) returns:
[{name, type, robot, movable_obj, region, start, end, occupy_start, occupy_end}], plus per-action
region offsets {name: (alpha, beta)} (relative to that action's own start) and durations {name: d}.

check_least_commitment(): for every pair of different robots' actions on the same region, compares
the start gap Tamer actually chose against derive_minimal_constraint()'s theoretical minimum -- the
evidence that Tamer's schedule forces two robots apart no more than the geometry requires, and
picks the cheaper of the two orders.

retime_schedule(): every constraint the schedule has to respect is a difference constraint
s_v - s_u >= w (same-robot sequencing, same-object handoffs, and the region constraint
s_j - s_i >= beta_ik - alpha_jk with Tamer's chosen order kept -- the symbolic plan never changes).
The earliest start times satisfying all of them are a longest-path solve from a time-zero source
(Bellman-Ford); a positive cycle means no re-timing exists and the caller must fall back to
re-planning (Step 5b).
"""
from collections import namedtuple

from mm_drrt.utils.minimal_temporal_constraint import derive_minimal_constraint

DEFAULT_TIGHT_TOLERANCE = 0.05  # Tamer separates mutex-ordered happenings by an epsilon of 0.01
SOURCE = '__time_zero__'

PairCheck = namedtuple('PairCheck', ['first', 'second', 'region', 'actual_gap', 'required_gap',
                                     'minimal_gap', 'minimal_order_chosen', 'occupancy_slack',
                                     'slack', 'reason'])
RetimeResult = namedtuple('RetimeResult', ['feasible', 'starts', 'makespan', 'original_makespan',
                                           'deltas', 'constraints'])


def offsets_by_action(schedule, by_type):
    """Expand per-action-type {'transit': (alpha, beta[, duration]), ...} into {action name:
    (alpha, beta)}. Action types missing from by_type (or None) are left out -- no region
    constraint for them."""
    return {a['name']: tuple(by_type[a['type']][:2]) for a in schedule
            if by_type.get(a['type']) is not None}


def durations_by_action(schedule, by_type=None):
    """{action name: duration}: by_type[action type] where given, else the solved end - start."""
    by_type = by_type or {}
    return {a['name']: float(by_type[a['type']]) if by_type.get(a['type']) is not None
            else a['end'] - a['start'] for a in schedule}


def region_pairs(schedule, offsets):
    """(first, second) action dicts for every pair of different robots' actions on the same region
    that both have offsets, ordered by which one the schedule lets into the region first."""
    pairs = []
    for i, a in enumerate(schedule):
        for b in schedule[i + 1:]:
            if a['robot'] == b['robot'] or a['region'] is None or a['region'] != b['region']:
                continue
            if a['name'] not in offsets or b['name'] not in offsets:
                continue
            first, second = (a, b) if a['occupy_start'] <= b['occupy_start'] else (b, a)
            pairs.append((first, second))
    return pairs


def _slack_reason(second, schedule, tol):
    """Why `second` starts later than the region constraint alone requires, if something else
    explains it: its own robot was still busy, the object it needs wasn't handed over yet, or it
    already starts at time zero."""
    if second['start'] <= tol:
        return 'time-zero'
    for other in schedule:
        if other is second or other['start'] >= second['start']:
            continue
        if abs(other['end'] - second['start']) > tol:
            continue
        if other['robot'] == second['robot']:
            return 'robot-sequence'
        if other['movable_obj'] is not None and other['movable_obj'] == second['movable_obj']:
            return 'object-handoff'
    return 'unexplained'


def _occupies_between(first, second, schedule, offsets):
    """True if another action enters the same region between `first` and `second` -- the pair
    isn't adjacent in the region's occupancy order, so its gap is a SUM of adjacent gaps (each of
    which is checked on its own), not a separation the region requires of this pair directly."""
    return any(k is not first and k is not second and k['region'] == second['region']
               and k['name'] in offsets
               and first['occupy_start'] <= k['occupy_start'] <= second['occupy_start']
               for k in schedule)


def check_least_commitment(schedule, offsets, tol=DEFAULT_TIGHT_TOLERANCE):
    """One PairCheck per region pair (see region_pairs):

      actual_gap: s_second - s_first as solved.
      required_gap: beta_first - alpha_second -- what the order Tamer chose actually needs.
      minimal_gap: derive_minimal_constraint()'s gap -- what the CHEAPER of the two orders needs.
      minimal_order_chosen: Tamer's order is (one of) the cheapest.
      occupancy_slack: occupy_start_second - occupy_end_first, from the occupy phases themselves.
      slack: actual_gap - max(required_gap, 0) -- extra delay beyond what the region requires.
      reason: 'tight' (slack <= tol), 'not-binding' (required_gap <= 0: they can't overlap even at
          equal start), or why the extra delay is explained by something else -- 'transitive'
          (another action occupies the region in between, so this gap is a sum of adjacent gaps),
          'robot-sequence', 'object-handoff', 'time-zero' -- else 'unexplained' (the schedule is
          NOT least-commitment for that pair).
    """
    checks = []
    for first, second in region_pairs(schedule, offsets):
        a_f, b_f = offsets[first['name']]
        a_s, b_s = offsets[second['name']]
        mc = derive_minimal_constraint(a_f, b_f, a_s, b_s)
        actual_gap = second['start'] - first['start']
        required_gap = b_f - a_s
        slack = actual_gap - max(required_gap, 0.0)
        if required_gap <= 0:
            reason = 'not-binding'
        elif slack <= tol:
            reason = 'tight'
        elif _occupies_between(first, second, schedule, offsets):
            reason = 'transitive'
        else:
            reason = _slack_reason(second, schedule, tol)
        checks.append(PairCheck(
            first=first['name'], second=second['name'], region=second['region'],
            actual_gap=actual_gap, required_gap=required_gap, minimal_gap=mc.gap,
            minimal_order_chosen=mc.order == 'i_before_j' or mc.raw_gap_i_before_j == mc.raw_gap_j_before_i,
            occupancy_slack=second['occupy_start'] - first['occupy_end'], slack=slack, reason=reason))
    return checks


def is_least_commitment(checks):
    return all(c.minimal_order_chosen and c.reason != 'unexplained' for c in checks)


def _precedence_constraints(schedule, durations):
    """Same-robot sequencing and cross-robot same-object handoffs, in the solved order: the
    successor may not start until the predecessor has finished (s_b - s_a >= d_a)."""
    constraints = []
    ordered = sorted(schedule, key=lambda a: a['start'])
    for key in ('robot', 'movable_obj'):
        last = {}
        for a in ordered:
            k = a[key]
            if k is None:
                continue
            if k in last and (key == 'robot' or last[k]['robot'] != a['robot']):
                prev = last[k]
                constraints.append((prev['name'], a['name'], durations[prev['name']], f'{key}-order'))
            last[k] = a
    return constraints


def _conflict_constraints(schedule, conflicts, forced_order=None):
    """Region constraints from motion-detected conflicts (motion_timing.detect_motion_conflicts):
    each is (name_i, name_j, alpha_i, beta_i, alpha_j, beta_j) for one colliding pair. The order is
    the one the solved schedule already had them entering their conflict windows in; a pair the
    schedule never ordered (equal entry) gets derive_minimal_constraint()'s cheaper order."""
    by_name = {a['name']: a for a in schedule}
    constraints = []
    # One order per pair: a pair can carry several conflict windows (e.g. its region window plus a
    # motion-detected one), and deciding each from its own entry time can point them in opposite
    # directions -- a cycle, i.e. a spuriously infeasible re-timing. The pair's first entry wins.
    # forced_order {frozenset((a, b)): name that goes first} overrides the solved order for a pair.
    first_of = dict(forced_order or {})
    for name_i, name_j, a_i, b_i, a_j, b_j in conflicts:
        key = frozenset((name_i, name_j))
        if key in first_of:
            winner = first_of[key]
            i_first = winner == name_i
        else:
            enter_i = by_name[name_i]['start'] + a_i
            enter_j = by_name[name_j]['start'] + a_j
            if enter_i == enter_j:
                i_first = derive_minimal_constraint(a_i, b_i, a_j, b_j).order == 'i_before_j'
            else:
                i_first = enter_i < enter_j
            first_of[key] = name_i if i_first else name_j
        if i_first:
            constraints.append((name_i, name_j, b_i - a_j, 'motion-conflict'))
        else:
            constraints.append((name_j, name_i, b_j - a_i, 'motion-conflict'))
    return constraints


def _region_constraints(schedule, offsets, order):
    """Region constraints for every pair sharing a region. order='solved' keeps the order the
    schedule already has; order='minimal' takes derive_minimal_constraint()'s cheaper order and
    order='maximal' the more expensive one (to show what the wrong choice costs) -- for
    two actions with no symbolic link (different robots and objects) the order on a shared region
    is a pure scheduling decision, which a satisficing planner like Tamer need not get right."""
    constraints = []
    for first, second in region_pairs(schedule, offsets):
        a_f, b_f = offsets[first['name']]
        a_s, b_s = offsets[second['name']]
        cheaper_is_second_first = derive_minimal_constraint(a_f, b_f, a_s, b_s).order == 'j_before_i'
        if (order == 'minimal' and cheaper_is_second_first) or \
                (order == 'maximal' and not cheaper_is_second_first and b_f - a_s != b_s - a_f):
            constraints.append((second['name'], first['name'], b_s - a_f, f'region ({order} order)'))
        else:
            constraints.append((first['name'], second['name'], b_f - a_s, 'region'))
    return constraints


def retime_schedule(schedule, offsets, durations, conflicts=None, max_makespan=None, order='solved',
                    extra=(), forced_order=None):
    """Earliest start times for the SAME actions, in the SAME per-robot/per-object order -- and,
    with order='solved', the same per-region order -- under new offsets/durations (e.g. measured on
    the executed trajectory).

    Args:
        schedule: the solved schedule (see module docstring).
        offsets: {name: (alpha, beta)} -- region pairs are constrained with these
            (s_second - s_first >= beta_first - alpha_second). Ignored when `conflicts` is given.
        durations: {name: d}.
        conflicts: optional motion-detected conflicts (see _conflict_constraints) replacing the
            region pairs: only pairs that really collide are constrained, and only over their
            conflict-derived intervals.
        max_makespan: optional deadline (s + d <= max_makespan for every action) -- the only way
            this problem becomes infeasible besides contradictory orders.
        order: 'solved' or 'minimal' (see _region_constraints). A 'minimal' choice that contradicts
            the same-robot/handoff orders is infeasible; the solved order is then used instead.
        extra: additional raw difference constraints [(u, v, w, kind)] meaning s_v - s_u >= w
            (e.g. motion_timing.async_schedule's alignment constraints for an idle robot).
        forced_order: {frozenset((a, b)): first} -- the order of a conflict pair, overriding the
            one read off the schedule (motion_timing.async_schedule flips a pair this way when its
            default order makes the re-timing infeasible).

    Returns RetimeResult: feasible, starts {name: s}, makespan, original_makespan (as solved),
    deltas {name: new s - solved s}, constraints [(pre, post, gap, kind)] used.
    """
    precedence = _precedence_constraints(schedule, durations) + list(extra)
    if conflicts is not None:
        return _solve_difference_constraints(schedule, durations, precedence +
                                             _conflict_constraints(schedule, conflicts, forced_order), max_makespan)
    result = _solve_difference_constraints(schedule, durations, precedence +
                                           _region_constraints(schedule, offsets, order), max_makespan)
    if not result.feasible and order != 'solved':
        result = _solve_difference_constraints(schedule, durations, precedence +
                                               _region_constraints(schedule, offsets, 'solved'), max_makespan)
    return result


def _solve_difference_constraints(schedule, durations, constraints, max_makespan):
    edges = [(SOURCE, a['name'], 0.0) for a in schedule]
    edges += [(u, v, w) for u, v, w, _ in constraints]
    if max_makespan is not None:
        edges += [(a['name'], SOURCE, durations[a['name']] - max_makespan) for a in schedule]

    # Longest path from SOURCE == earliest times satisfying every s_v - s_u >= w; a positive cycle
    # (Bellman-Ford still relaxing after |V| - 1 rounds) means the constraints contradict.
    dist = {SOURCE: 0.0}
    nodes = [SOURCE] + [a['name'] for a in schedule]
    for _ in range(len(nodes) - 1):
        changed = False
        for u, v, w in edges:
            if u in dist and dist[u] + w > dist.get(v, float('-inf')) + 1e-12:
                dist[v] = dist[u] + w
                changed = True
        if not changed:
            break
    feasible = dist[SOURCE] <= 1e-9 and not any(
        u in dist and dist[u] + w > dist.get(v, float('-inf')) + 1e-9 for u, v, w in edges)

    original_makespan = max(a['end'] for a in schedule)
    if not feasible:
        return RetimeResult(False, None, None, original_makespan, None, constraints)
    starts = {a['name']: dist[a['name']] for a in schedule}
    makespan = max(starts[n] + durations[n] for n in starts)
    deltas = {a['name']: starts[a['name']] - a['start'] for a in schedule}
    return RetimeResult(True, starts, makespan, original_makespan, deltas, constraints)


if __name__ == '__main__':
    # Self-check, matching minimal_temporal_constraint.py's convention (no pytest in this repo).
    def act(name, robot, obj, region, start, end, occ=None):
        occ = occ or (start, end)
        return {'name': name, 'type': 'transfer', 'robot': robot, 'movable_obj': obj, 'region': region,
                'start': start, 'end': end, 'occupy_start': occ[0], 'occupy_end': occ[1]}

    # Two robots placing onto one pad, each occupying [4, 6] of a 10s action: the region needs
    # s_j - s_i >= 6 - 4 = 2. Tamer solved it with exactly that gap (+ epsilon) -> tight.
    sched = [act('a', 'r0', 'o0', 'pad', 0.0, 10.0, (4.0, 6.0)),
             act('b', 'r1', 'o1', 'pad', 2.01, 12.01, (6.01, 8.01))]
    offs = {'a': (4.0, 6.0), 'b': (4.0, 6.0)}
    [c] = check_least_commitment(sched, offs)
    assert c.reason == 'tight' and c.minimal_order_chosen and abs(c.minimal_gap - 2.0) < 1e-9, c
    assert is_least_commitment([c])

    # Same pair, but b waits until a fully finishes (full serialization): unexplained slack.
    sched_serial = [sched[0], act('b', 'r1', 'o1', 'pad', 10.0, 20.0, (14.0, 16.0))]
    [c] = check_least_commitment(sched_serial, offs)
    assert c.reason == 'unexplained' and abs(c.slack - 8.0) < 1e-9, c
    assert not is_least_commitment([c])

    # ...unless b's own robot was busy until then: the slack is explained, not a region artifact.
    sched_busy = sched_serial + [act('p', 'r1', 'o2', 'zone', 0.0, 10.0)]
    [c] = check_least_commitment(sched_busy, offs)
    assert c.reason == 'robot-sequence', c

    # Three robots chained through one pad: a -> b -> c are each tight, and a -> c's larger gap is
    # just the sum of the two -- transitive, not slack.
    sched_chain = [act('a', 'r0', 'o0', 'pad', 0.0, 10.0, (4.0, 6.0)),
                   act('b', 'r1', 'o1', 'pad', 2.01, 12.01, (6.01, 8.01)),
                   act('c', 'r2', 'o2', 'pad', 4.02, 14.02, (8.02, 10.02))]
    checks = check_least_commitment(sched_chain, dict(offs, c=(4.0, 6.0)))
    assert [c.reason for c in checks] == ['tight', 'transitive', 'tight'], checks
    assert is_least_commitment(checks)

    # Tamer picked the expensive order: a occupies [0, 10], b only [8, 10]; b-first needs 10,
    # a-first only 2.
    offs2 = {'a': (0.0, 10.0), 'b': (8.0, 10.0)}
    sched2 = [act('b', 'r1', 'o1', 'pad', 0.0, 10.0, (8.0, 10.0)),
              act('a', 'r0', 'o0', 'pad', 0.0, 10.0, (10.01, 20.01))]
    [c] = check_least_commitment(sched2, offs2)
    assert c.first == 'b' and not c.minimal_order_chosen and abs(c.minimal_gap - 2.0) < 1e-9, c

    # ...and re-timing with order='minimal' swaps that pair: a goes first, only 2s later than b's
    # start would have needed, instead of a waiting the full 10s.
    durs2 = {'a': 10.0, 'b': 10.0}
    r = retime_schedule(sched2, offs2, durs2)
    assert r.feasible and abs(r.starts['a'] - 10.0) < 1e-9 and r.starts['b'] == 0.0, r
    r = retime_schedule(sched2, offs2, durs2, order='minimal')
    assert r.feasible and r.starts['a'] == 0.0 and abs(r.starts['b'] - 2.0) < 1e-9, r
    assert abs(r.makespan - 12.0) < 1e-9
    # order='maximal' always takes the expensive order, whichever one Tamer happened to choose.
    assert abs(retime_schedule(sched2, offs2, durs2, order='maximal').makespan - 20.0) < 1e-9
    r = retime_schedule(sched2, offs2, durs2, order='minimal')
    flipped = [dict(x, start=r.starts[x['name']], occupy_start=r.starts[x['name']] + offs2[x['name']][0])
               for x in sched2]
    assert abs(retime_schedule(flipped, offs2, durs2, order='maximal').makespan - 20.0) < 1e-9

    # Re-timing with WIDER measured offsets: b must move from 2.01 to 6 - 1 = 5.
    durs = {'a': 10.0, 'b': 10.0}
    r = retime_schedule(sched, {'a': (1.0, 6.0), 'b': (1.0, 6.0)}, durs)
    assert r.feasible and r.starts['a'] == 0.0 and abs(r.starts['b'] - 5.0) < 1e-9, r
    assert abs(r.makespan - 15.0) < 1e-9 and abs(r.deltas['b'] - (5.0 - 2.01)) < 1e-9, r

    # Re-timing can also pull actions EARLIER when the measured window is narrower.
    r = retime_schedule(sched_serial, offs, durs)
    assert r.feasible and abs(r.starts['b'] - 2.0) < 1e-9, r

    # Same-robot sequencing is kept: r0's second action can't start before its first ends.
    sched3 = sched + [act('c', 'r0', 'o2', 'zone', 10.0, 20.0)]
    r = retime_schedule(sched3, offs, dict(durs, c=10.0))
    assert r.feasible and r.starts['c'] >= 10.0 - 1e-9, r

    # A makespan deadline the widened window can't meet: infeasible -> caller re-plans.
    r = retime_schedule(sched, {'a': (1.0, 6.0), 'b': (1.0, 6.0)}, durs, max_makespan=12.0)
    assert not r.feasible and r.starts is None, r

    # Motion conflicts replace region pairs: a pair that never collides is left unconstrained.
    r = retime_schedule(sched, offs, durs, conflicts=[])
    assert r.feasible and r.starts['b'] == 0.0, r
    r = retime_schedule(sched, offs, durs, conflicts=[('a', 'b', 5.0, 6.0, 5.5, 7.0)])
    assert r.feasible and abs(r.starts['b'] - 0.5) < 1e-9, r

    # Two windows for one pair are ordered the same way even if their own entries disagree.
    r = retime_schedule(sched, offs, durs, conflicts=[('a', 'b', 4.0, 6.0, 4.0, 6.0), ('a', 'b', 9.0, 9.5, 0.0, 1.0)])
    assert r.feasible and r.starts['b'] >= r.starts['a'], r

    # Contradictory orders: a hands o0 over to b (s_b - s_a >= 10), but a's conflict window [15, 16]
    # starts after b's [0, 20] does in the solved schedule, so b goes first there (s_a - s_b >= 5).
    # Positive cycle -> no re-timing exists without changing the plan -> infeasible.
    sched4 = [act('a', 'r0', 'o0', 'pad', 0.0, 10.0), act('b', 'r1', 'o0', 'pad', 10.0, 20.0)]
    r = retime_schedule(sched4, {}, durs, conflicts=[('a', 'b', 15.0, 16.0, 0.0, 20.0)])
    assert not r.feasible, r
    assert retime_schedule(sched4, {}, durs, conflicts=[]).feasible

    print("All schedule_repair self-checks passed.")
