"""C1 + C2 from the motion itself: how long a refined trajectory actually takes, when it is inside
its coordination region, and which pairs of different robots' actions really collide.

Timing model: a piecewise-linear joint-space path executed as fast as the joint velocity limits
allow, i.e. each step takes max_j |dq_j| / vmax_j. Unlike trajectory_region_fractions()'s
waypoint-index fractions, this does not depend on how densely a planner happened to interpolate a
segment -- within one dRRT* composite node the two arms routinely carry different waypoint counts
(e.g. 30 vs. 19), so an index is not a time.

Executed timeline: robots move simultaneously along each composite node's edge and synchronize at
the node (the node takes as long as its slowest robot's segment) -- the same lock-step
replay_composite_path() plays back.
"""
from collections import namedtuple

import numpy as np

from mm_drrt.utils import rai_utils as ru
from mm_drrt.utils.coordination_regions import region_volume_from_frame, point_in_region

# Franka Emika Panda joint velocity limits (rad/s), joints 1-7 (datasheet values).
FRANKA_VELOCITY_LIMITS = (2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61)
DEFAULT_VELOCITY_LIMIT = 1.0  # rad/s, for any non-7-DoF arm (no per-joint data here)

MotionConflict = namedtuple('MotionConflict', ['name_i', 'name_j', 'alpha_i', 'beta_i',
                                               'alpha_j', 'beta_j', 'region_i', 'region_j', 'hits'])


def velocity_limits(arm_joints, scale=1.0):
    base = FRANKA_VELOCITY_LIMITS if len(arm_joints) == len(FRANKA_VELOCITY_LIMITS) \
        else (DEFAULT_VELOCITY_LIMIT,) * len(arm_joints)
    return np.asarray(base, dtype=float) * scale


def path_timestamps(path, vlim):
    """Cumulative time at each waypoint (first is 0.0). Only the last len(vlim) values of each
    waypoint are the arm's joints (same convention as gripper_positions_along_path)."""
    if not path:
        return []
    q = np.asarray([np.asarray(p, dtype=float)[-len(vlim):] for p in path])
    steps = np.max(np.abs(np.diff(q, axis=0)) / vlim, axis=1) if len(q) > 1 else np.zeros(0)
    return [0.0] + [float(t) for t in np.cumsum(steps)]


def trajectory_region_times(robot, arm_joints, gripper_frame, path, region, vlim):
    """(t_entry, t_exit, total_time) along `path`: when the gripper first/last is inside `region`
    (a coordination_regions.RegionVolume), in seconds under the velocity-limited timing model.
    t_entry/t_exit are None if the gripper never enters it."""
    from mm_drrt.utils.rai_motion_planner_utils import gripper_positions_along_path
    times = path_timestamps(path, vlim)
    if not times:
        return None, None, 0.0
    positions = gripper_positions_along_path(robot, arm_joints, gripper_frame, path)
    inside = [i for i, p in enumerate(positions) if point_in_region(p, region)]
    if not inside:
        return None, None, float(times[-1])
    return float(times[inside[0]]), float(times[inside[-1]]), float(times[-1])


def split_at_carry(path, carry_conf, tol=1e-6):
    """[(i0, i1)] index ranges of the motions between visits to carry_conf: every transit/transfer
    starts at carry_conf and arm_retrieval_motion always ends it there (an exact match, see
    DurationConflictRaiEnvironment.measure_executed_region_offsets). The final segment may end
    without returning to carry_conf. Segments with no motion at all are dropped."""
    carry = np.asarray(carry_conf, dtype=float)
    at_carry = [np.linalg.norm(np.asarray(q, dtype=float)[-len(carry):] - carry) < tol for q in path]
    segments, start = [], 0
    for i in range(1, len(path)):
        if at_carry[i] and not at_carry[i - 1]:
            segments.append((start, i))
        if at_carry[i]:
            start = i
    if start < len(path) - 1 and not all(at_carry[start:]):
        segments.append((start, len(path) - 1))
    return segments


def split_at_events(path, carry_conf, events, tol=1e-6):
    """[(i0, i1)] index ranges, one per action, for a path whose actions each END with an
    attachment change (a transit's grasp, a transfer's release) followed by the retrieval back to
    carry_conf. Action k ends at the first carry_conf visit at or after events[k]; the next action
    starts where the robot last sits at carry_conf before moving again. Unlike split_at_carry this
    survives a path that passes through carry_conf mid-action (observed with 4 arms). Returns None
    if an event never reaches carry_conf (the caller then falls back to split_at_carry)."""
    carry = np.asarray(carry_conf, dtype=float)
    at_carry = [np.linalg.norm(np.asarray(q, dtype=float)[-len(carry):] - carry) < tol for q in path]
    segments, start = [], 0
    for k, e in enumerate(events):
        while start + 1 < len(path) and at_carry[start] and at_carry[start + 1]:
            start += 1
        end = next((i for i in range(max(e, start + 1), len(path)) if at_carry[i]), None)
        if end is None:
            if k != len(events) - 1:
                return None
            end = len(path) - 1
        segments.append((start, end))
        start = end
    return segments


def executed_action_timeline(env, composite_path, action_orders, plan, vel_scale=1.0):
    """Per-action timing of the EXECUTED composite path, keyed by plan action name:
    {name: {robot_index, type, region, path, start, end, duration, motion_times,
            entry, exit}}
    start/end are absolute (lock-step) times; duration and entry/exit (region occupancy, relative to
    the action's own start, None if the gripper never entered it) are in motion time -- the
    velocity-limited time of the action's own trajectory, excluding waits for the other robots.

    A robot whose motion segments can't be matched one-to-one with its non-'return' actions is
    skipped with a warning (best-effort, like measure_executed_region_offsets)."""
    robots = list(env.robots.values())
    specs = [ru._spec_of(r) for r in robots]
    vlims = [velocity_limits(s[0], vel_scale) for s in specs]

    full = [[] for _ in robots]
    abs_times = [[] for _ in robots]
    # Index (in full[r]) of every attachment change: a node's attachments describe its LAST
    # waypoint (see replay_composite_path), so that is where a grasp/release happens.
    events = [[] for _ in robots]
    held = [None for _ in robots]
    node_ends = [[] for _ in robots]  # (last waypoint index of a node, that node's subprob_id[r])
    t_node = 0.0
    for node in composite_path:
        node_end = t_node
        for r in range(len(robots)):
            seg = node.sub_local_paths[r] if node.sub_local_paths and len(node.sub_local_paths) > r else []
            if not seg:
                continue
            prev = [full[r][-1]] if full[r] else []
            ts = path_timestamps(prev + list(seg), vlims[r])[len(prev):]
            full[r].extend(seg)
            abs_times[r].extend(t_node + t for t in ts)
            node_end = max(node_end, t_node + (ts[-1] if ts else 0.0))
            now = node.attachments[r] if node.attachments and len(node.attachments) > r else None
            if now != held[r]:
                events[r].append(len(full[r]) - 1)
                held[r] = now
            if node.subprob_id and len(node.subprob_id) > r:
                node_ends[r].append((len(full[r]) - 1, node.subprob_id[r]))
        t_node = node_end

    timeline = {}
    for r, robot in enumerate(robots):
        arm_joints, gripper_frame, _, carry_conf = specs[r]
        names = [n for n in action_orders.get(robot, ()) if plan[n][0] != 'return']
        segments = split_at_events(full[r], carry_conf, events[r]) \
            if len(events[r]) == len(names) else None
        if segments is None:
            segments = split_at_carry(full[r], carry_conf)
        if len(segments) != len(names):
            print(f"Warning: robot {r}: {len(segments)} motion segments vs. {len(names)} actions "
                  f"{names} -- skipping its executed timeline.")
            continue
        for k, (name, (i0, i1)) in enumerate(zip(names, segments)):
            a_type, _, m_obj, _, region_frame = plan[name]
            in_segment = [e for e in events[r] if i0 <= e <= i1]
            path = full[r][i0:i1 + 1]
            region = region_volume_from_frame(ru._config_of(robot), region_frame)
            entry, exit_, duration = trajectory_region_times(robot, arm_joints, gripper_frame, path,
                                                            region, vlims[r])
            timeline[name] = {
                'robot_index': r, 'type': a_type, 'region': region_frame, 'path': path,
                'start': abs_times[r][i0], 'end': abs_times[r][i1], 'duration': duration,
                'motion_times': path_timestamps(path, vlims[r]), 'entry': entry, 'exit': exit_,
                # Where in `path` the grasp (transit) / release (transfer) happens, and of what.
                'event_index': _contact_index(node_ends[r], k, i0, i1,
                                              (in_segment[0] - i0) if in_segment else len(path) - 1),
                'path_offset': i0,
                # Absolute (lock-step) time of every waypoint of `path`.
                'abs_times': abs_times[r][i0:i1 + 1],
                'obj': m_obj,
            }
    return timeline


def collapse_by_type(timeline):
    """{'transit': (alpha, beta, duration) or None, 'transfer': ...}: union over the actions of
    each type (min entry, max exit, max duration) -- the same domain-wide-per-action-type
    collapse measure_region_offsets() uses."""
    out = {}
    for a_type in ('transit', 'transfer'):
        acts = [a for a in timeline.values() if a['type'] == a_type]
        inside = [a for a in acts if a['entry'] is not None]
        if not inside:
            out[a_type] = None
            continue
        out[a_type] = (min(a['entry'] for a in inside), max(a['exit'] for a in inside),
                       max(a['duration'] for a in acts))
    return out


def _subsample(n, max_samples):
    if not max_samples or n <= max_samples:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, max_samples).round().astype(int).tolist()))


def detect_motion_conflicts(env, timeline, max_samples=None):
    """C2: which pairs of different robots' actions REALLY collide if run concurrently, and over
    which part of each action. For every such pair, sweeps subsampled waypoint pairs (q_i, q_j)
    (every waypoint by default -- the colliding window can be short enough that uniform subsampling
    skips it entirely) through dRRT*'s own get_inter_robots_collision_fn -- a time-alignment-free test: it asks
    whether ANY alignment of the two motions could collide, which is what a mutex has to rule out.
    Every robot not in the pair is held at its carry_conf.

    Returns [MotionConflict] for colliding pairs only. alpha/beta are motion-time offsets into each
    action, widened to the neighbouring samples around the first/last colliding one (so
    subsampling never shrinks the window). Known approximation: attached objects are not moved
    with the gripper during the sweep, so a carried block's own volume is not checked."""
    from mm_drrt.utils.rai_motion_planner_utils import get_inter_robots_collision_fn
    robots = list(env.robots.values())
    joints = env.get_joints(robots)
    collision_fn = get_inter_robots_collision_fn(robots, joints, num_robots=len(robots))
    carries = [list(ru._spec_of(r)[3]) for r in robots]

    names = list(timeline)
    conflicts = []
    saved_world = env.save_world()
    try:
        for x, name_i in enumerate(names):
            for name_j in names[x + 1:]:
                a, b = timeline[name_i], timeline[name_j]
                ri, rj = a['robot_index'], b['robot_index']
                if ri == rj:
                    continue
                si = _subsample(len(a['path']), max_samples)
                sj = _subsample(len(b['path']), max_samples)
                hit_i, hit_j, hits = set(), set(), 0
                for pi, i in enumerate(si):
                    for pj, j in enumerate(sj):
                        q = list(carries)
                        q[ri] = list(a['path'][i])[-len(joints[ri]):]
                        q[rj] = list(b['path'][j])[-len(joints[rj]):]
                        colliding = collision_fn(q, mode='index')
                        if ri in colliding and rj in colliding:
                            hit_i.add(pi)
                            hit_j.add(pj)
                            hits += 1
                if not hits:
                    continue
                ti = [a['motion_times'][i] for i in si]
                tj = [b['motion_times'][j] for j in sj]
                conflicts.append(MotionConflict(
                    name_i, name_j,
                    float(ti[max(min(hit_i) - 1, 0)]), float(ti[min(max(hit_i) + 1, len(ti) - 1)]),
                    float(tj[max(min(hit_j) - 1, 0)]), float(tj[min(max(hit_j) + 1, len(tj) - 1)]),
                    a['region'], b['region'], hits))
    finally:
        env.restore_world(saved_world)
    return conflicts


def _contact_index(node_ends, k, i0, i1, fallback):
    """Waypoint (within the action's segment [i0, i1]) where the grasp/release really happens: the
    end of the robot's approach motion. Each action is 3 roadmap segments (base, approach,
    retrieval), so action k's approach is subproblem 3k + 1, and dRRT* moves the robot past it
    exactly when it reaches the approach goal -- the grasp/place configuration. Contact is
    therefore the last waypoint of the first node whose subprob_id is >= 3k + 2 (checked against
    the planned grasp_conf: an exact match on every action of the relay and the 3-arm round
    table).

    Two earlier choices were wrong: the node where node.attachments flips can end a few waypoints
    before the place pose (the arm then 'placed' an already-released block), and the gripper's
    lowest point inside the region is not the grasp pose for tall boxes in a large region (the box
    was yanked 9 cm to the gripper). Falls back to `fallback` if no such node lies in the segment."""
    for idx, subprob in node_ends:
        if subprob >= 3 * k + 2:
            return idx - i0 if i0 <= idx <= i1 else fallback
    return fallback


def contact_events(timeline):
    """{robot index: {waypoint index in that robot's concatenated composite path: (kind, obj)}}
    -- the grasp/release of every timed action, for replay_composite_path(events=...)."""
    events = {}
    for a in timeline.values():
        kind = 'grasp' if a['type'] == 'transit' else 'release'
        events.setdefault(a['robot_index'], {})[a['path_offset'] + a['event_index']] = (kind, a['obj'])
    return events


def lockstep_makespan(timeline):
    """Makespan of the composite path executed SYNCHRONOUSLY: every robot waits at each dRRT*
    composite node for the slowest one (executed_action_timeline's absolute times)."""
    return max(a['end'] for a in timeline.values())


AsyncEvent = namedtuple('AsyncEvent', ['time', 'robot_index', 'kind', 'obj', 'action'])


class AsyncExecution(object):
    """Each robot follows its OWN refined paths on its own clock: action `name` starts at
    starts[name] and runs at velocity-limited speed (its timeline motion_times); between actions
    the robot waits where its last action ended (carry_conf). No robot waits for another robot's
    progress through dRRT*'s composite nodes -- only for what the schedule says."""

    def __init__(self, timeline, starts, num_robots):
        self.timeline = timeline
        self.starts = dict(starts)
        self.per_robot = [sorted((n for n in timeline if timeline[n]['robot_index'] == r),
                                 key=lambda n: self.starts[n]) for r in range(num_robots)]
        self.makespan = max(self.starts[n] + timeline[n]['duration'] for n in timeline)
        self.events = sorted(
            (AsyncEvent(self.starts[n] + timeline[n]['motion_times'][timeline[n]['event_index']],
                        timeline[n]['robot_index'], 'grasp' if timeline[n]['type'] == 'transit' else 'release',
                        timeline[n]['obj'], n) for n in timeline),
            key=lambda e: e.time)

    def active(self, r, t):
        """(action name, local time) robot r is executing at time t, or (None, None) if idle."""
        for n in self.per_robot[r]:
            local = t - self.starts[n]
            if 0.0 <= local <= self.timeline[n]['duration']:
                return n, local
        return None, None

    def conf(self, r, t):
        """Robot r's arm configuration at time t (linear interpolation along its path)."""
        names = self.per_robot[r]
        if not names:
            return None
        current = names[0]
        for n in names:
            if self.starts[n] <= t:
                current = n
        a = self.timeline[current]
        local = min(max(t - self.starts[current], 0.0), a['duration'])
        times, path = a['motion_times'], a['path']
        k = int(np.searchsorted(times, local, side='right')) - 1
        k = min(max(k, 0), len(path) - 1)
        if k == len(path) - 1 or times[k + 1] <= times[k]:
            return list(path[k])
        w = (local - times[k]) / (times[k + 1] - times[k])
        return list((1 - w) * np.asarray(path[k], dtype=float) + w * np.asarray(path[k + 1], dtype=float))


def verify_async(env, execution, dt=0.02, tolerance=None):
    """Sweeps the asynchronous execution on a dt grid for collisions. Shifting robots relative to
    each other leaves the timing dRRT* verified, so this is what makes an asynchronous schedule
    safe to run. Blocks are part of the sweep: each starts at its initial pose, is attached to its
    robot's gripper at that robot's grasp event and rests on its destination after the release
    event. A robot touching a block another robot is carrying counts as a collision between the
    two robots; touching a resting block it isn't holding is reported with that block.

    Returns None if collision-free, else (t, robot_indices, {robot_index: active action or None},
    resting_block or None)."""
    from mm_drrt.utils.rai_motion_planner_utils import get_inter_robots_collision_fn
    robots = list(env.robots.values())
    joints = env.get_joints(robots)
    tolerance = ru.COLLISION_TOLERANCE if tolerance is None else tolerance
    collision_fn = get_inter_robots_collision_fn(robots, joints, num_robots=len(robots))
    carries = [list(ru._spec_of(r)[3]) for r in robots]
    prefixes = [getattr(getattr(r, 'spec', None), 'frame_prefix', None) for r in robots]
    grippers = [ru._spec_of(r)[1] for r in robots]
    C = ru._config_of(robots[0])
    blocks = list(env.m_objs)
    destinations = {(e.robot_index, e.obj): execution.timeline[e.action]['region'] for e in execution.events
                    if e.kind == 'release'}

    def owner_of_frame(name):
        for r, pre in enumerate(prefixes):
            if pre and name.startswith(pre):
                return r
        return None

    saved_world = env.save_world()
    try:
        for obj in blocks:
            env.m_objs_init_placements[obj].assign()
        holder = {}
        pending = list(execution.events)
        for t in np.arange(0.0, execution.makespan + dt, dt):
            q = [execution.conf(r, t) or carries[r] for r in range(len(robots))]
            q = [list(qr)[-len(joints[r]):] for r, qr in enumerate(q)]
            colliding = collision_fn(q, mode='index')  # also sets every robot's joints
            while pending and pending[0].time <= t:
                e = pending.pop(0)
                if e.kind == 'grasp':
                    C.attach(grippers[e.robot_index], e.obj)
                    holder[e.obj] = e.robot_index
                else:
                    dest = destinations[(e.robot_index, e.obj)]
                    C.attach(dest, e.obj)
                    f, d = C.getFrame(e.obj), C.getFrame(dest)
                    pos = f.getPosition()
                    f.setPosition([pos[0], pos[1], d.getPosition()[2] + d.getSize()[2] / 2.0 + f.getSize()[2] / 2.0])
                    f.setQuaternion([1.0, 0.0, 0.0, 0.0])
                    holder.pop(e.obj, None)
            if len(colliding) >= 2:
                return float(t), [int(c) for c in colliding], \
                    {int(c): execution.active(int(c), t)[0] for c in colliding}, None
            C.computeCollisions()
            for x, y, pen in C.getCollisions():
                if pen >= -tolerance:
                    continue
                for robot_frame, obj in ((x, y), (y, x)):
                    r = owner_of_frame(robot_frame)
                    if r is None or obj not in blocks or holder.get(obj) == r:
                        continue
                    if obj in holder:  # carried by another robot: a robot-robot collision
                        pair = [r, holder[obj]]
                        return float(t), pair, {k: execution.active(k, t)[0] for k in pair}, None
                    return float(t), [r], {r: execution.active(r, t)[0]}, obj
    finally:
        env.restore_world(saved_world)
    return None


AsyncResult = namedtuple('AsyncResult', ['feasible', 'execution', 'starts', 'makespan', 'rounds',
                                         'conflicts', 'log'])


def _idle_alignment(env, timeline, mover, idle_robot, step=0.04):
    """An idle robot (waiting where its last action ended, normally carry_conf) is in `mover`'s
    way. dRRT*'s lock-step composite path is collision-free, so when `mover` passed that spot, the
    idle robot was busy with one of its own actions k. Keep a relation like that one: search the
    offset o = s_k - s_mover on a `step` grid and keep those for which, at every instant of
    mover's colliding window [u, v], the idle robot is inside k and its time-aligned pose clears
    mover's. Each contiguous range [lo, hi] of safe offsets is two difference constraints
    (s_k - s_a >= lo, s_a - s_k >= -hi), so re-timing stays a shortest-path solve.

    Returns a list of such constraint pairs -- the range containing the lock-step offset first, then
    the others by distance to it -- or None if none exists."""
    from mm_drrt.utils.rai_motion_planner_utils import get_inter_robots_collision_fn
    robots = list(env.robots.values())
    joints = env.get_joints(robots)
    collision_fn = get_inter_robots_collision_fn(robots, joints, num_robots=len(robots))
    carries = [list(ru._spec_of(r)[3]) for r in robots]
    a = timeline[mover]
    ra = a['robot_index']
    rest = [timeline[n]['path'][-1] for n in timeline if timeline[n]['robot_index'] == idle_robot] or \
        [carries[idle_robot]]

    def hits(qa, qr):
        q = list(carries)
        q[ra] = list(qa)[-len(joints[ra]):]
        q[idle_robot] = list(qr)[-len(joints[idle_robot]):]
        c = collision_fn(q, mode='index')
        return ra in c and idle_robot in c

    saved_world = env.save_world()
    try:
        bad = [i for i, qa in enumerate(a['path']) if any(hits(qa, qr) for qr in (carries[idle_robot], rest[-1]))]
        if not bad:
            return None
        window = range(max(bad[0] - 1, 0), min(bad[-1] + 1, len(a['path']) - 1) + 1)
        ta = a['motion_times']
        options = []
        for k in (n for n in timeline if timeline[n]['robot_index'] == idle_robot):
            b = timeline[k]
            single = AsyncExecution({k: b}, {k: 0.0}, len(robots))
            lo_o, hi_o = ta[window[-1]] - b['duration'], ta[window[0]]
            safe = []
            for o in np.arange(lo_o, hi_o + 1e-9, step):
                safe.append((o, all(0.0 <= ta[i] - o <= b['duration'] and
                                    not hits(a['path'][i], single.conf(idle_robot, ta[i] - o)) for i in window)))
            # The lock-step offset between the two actions: their absolute starts there.
            o_lock = b['start'] - a['start']
            run = []
            for o, ok in safe + [(None, False)]:
                if ok:
                    run.append(o)
                elif run:
                    lo, hi = run[0], run[-1]
                    dist = 0.0 if lo <= o_lock <= hi else min(abs(lo - o_lock), abs(hi - o_lock))
                    options.append((dist, [(mover, k, lo, 'idle alignment'), (k, mover, -hi, 'idle alignment')]))
                    run = []
        options.sort(key=lambda x: x[0])
        return [o for _, o in options] or None
    finally:
        env.restore_world(saved_world)


def async_schedule(env, schedule, timeline, conflicts=None, max_rounds=8, dt=0.02):
    """Asynchronous re-timing + replanning loop (Step 5a, then verification):

      1. re-time the solved schedule (schedule_repair.retime_schedule: same actions, same orders)
         with each action's MEASURED motion duration and its own measured region occupancy -- or,
         if motion conflicts are given, only those really-colliding pairs over their conflict
         intervals;
      2. build the asynchronous execution and sweep it for robot-robot collisions (verify_async);
      3. on a collision between two active actions, find that pair's conflict interval
         (detect_motion_conflicts restricted to the pair), add it as a constraint and re-time again.

    Pair orders come from dRRT*'s lock-step execution (the timeline's absolute starts).

    Returns AsyncResult(feasible, execution, starts, makespan, rounds, conflicts, log); feasible is
    False if re-timing became infeasible, a collision involved an idle robot (no action to
    constrain), or max_rounds ran out -- the caller then falls back to lock-step execution or to
    re-planning (Step 5b)."""
    from mm_drrt.utils.schedule_repair import retime_schedule

    durations = {n: timeline[n]['duration'] for n in timeline}
    # Orders (per robot, per handed-over object, per conflicting pair) are read off dRRT*'s
    # verified lock-step execution, not Tamer's schedule: it is one consistent timeline, and the
    # idle-robot alignment below is anchored to it too -- mixing the two sources of order produced
    # contradictory (cyclic, infeasible) constraints with 3 arms.
    sched = [dict(a, start=timeline[a['name']]['start'], end=timeline[a['name']]['start'] + durations[a['name']])
             for a in schedule if a['name'] in timeline]
    if conflicts is None:
        conflicts = []
        for x, a in enumerate(sched):
            for b in sched[x + 1:]:
                ta, tb = timeline[a['name']], timeline[b['name']]
                if ta['robot_index'] == tb['robot_index'] or ta['region'] != tb['region']:
                    continue
                if ta['entry'] is None or tb['entry'] is None:
                    continue
                conflicts.append((a['name'], b['name'], ta['entry'], ta['exit'], tb['entry'], tb['exit']))
    conflicts = list(conflicts)
    extra = []
    log = []
    for rounds in range(1, max_rounds + 1):
        result = retime_schedule(sched, {}, durations, conflicts=conflicts, extra=extra)
        if not result.feasible:
            log.append(f"round {rounds}: re-timing infeasible")
            return AsyncResult(False, None, None, None, rounds, conflicts, log)
        execution = AsyncExecution(timeline, result.starts, len(env.robots))
        hit = verify_async(env, execution, dt=dt)
        if hit is None:
            log.append(f"round {rounds}: makespan {execution.makespan:.3f}, collision-free")
            return AsyncResult(True, execution, result.starts, execution.makespan, rounds, conflicts, log)
        t, robots_hit, active, resting = hit
        names = [active[r] for r in robots_hit]
        if resting is not None:
            log.append(f"round {rounds}: makespan {execution.makespan:.3f}, robot {robots_hit[0]} ({names[0]}) "
                       f"hits {resting} resting at t={t:.2f}s -- re-timing cannot move a resting block")
            return AsyncResult(False, execution, result.starts, execution.makespan, rounds, conflicts, log)
        log.append(f"round {rounds}: makespan {execution.makespan:.3f}, collision at t={t:.2f}s "
                   f"between robots {robots_hit} ({names})")
        movers = [n for n in names if n is not None]
        if len(robots_hit) != 2 or not movers:
            log.append("  no moving action to constrain")
            return AsyncResult(False, execution, result.starts, execution.makespan, rounds, conflicts, log)
        if len(movers) == 2:
            pair = {n: timeline[n] for n in movers}
            found = detect_motion_conflicts(env, pair)
            if not found:
                # The sweep saw a collision the pairwise check didn't (interpolation between
                # waypoints): fall back to serializing the pair's whole actions.
                a, b = movers
                found = [MotionConflict(a, b, 0.0, timeline[a]['duration'], 0.0, timeline[b]['duration'],
                                        timeline[a]['region'], timeline[b]['region'], 0)]
            for c in found:
                conflicts.append((c.name_i, c.name_j, c.alpha_i, c.beta_i, c.alpha_j, c.beta_j))
        else:
            idle = next(r for r in robots_hit if active[r] is None)
            options = _idle_alignment(env, timeline, movers[0], idle) or []
            chosen = next((o for o in options if retime_schedule(sched, {}, durations, conflicts=conflicts,
                                                                 extra=extra + o).feasible), None)
            if chosen is None:
                log.append(f"  robot {idle} idles in {movers[0]}'s way and none of its actions can be "
                           f"aligned to clear it")
                return AsyncResult(False, execution, result.starts, execution.makespan, rounds, conflicts, log)
            extra += chosen
            log.append(f"  robot {idle} idles in {movers[0]}'s way: time {movers[0]} against {chosen[0][1]} "
                       f"(offset in [{chosen[0][2]:.2f}, {-chosen[1][2]:.2f}]s"
                       f"{', as in dRRT*' + chr(39) + 's lock-step solution' if chosen is options[0] else ''})")
        for a in sched:
            a['start'] = result.starts[a['name']]
            a['end'] = a['start'] + durations[a['name']]
    log.append(f"gave up after {max_rounds} rounds")
    return AsyncResult(False, None, None, None, max_rounds, conflicts, log)


if __name__ == '__main__':
    # Pure-function self-checks (the FK/collision parts need a live RAI scene -- exercised via
    # main_rai.py instead).
    vlim = velocity_limits(['j'] * 7)
    assert vlim[0] == 2.175 and vlim[6] == 2.61
    # One step moving joint 0 by 2.175 rad takes 1s regardless of how finely it is interpolated.
    coarse = [[0.0] * 7, [2.175] + [0.0] * 6]
    fine = [[x] + [0.0] * 6 for x in np.linspace(0.0, 2.175, 11)]
    assert abs(path_timestamps(coarse, vlim)[-1] - 1.0) < 1e-9
    assert abs(path_timestamps(fine, vlim)[-1] - 1.0) < 1e-9
    # The slowest joint relative to its limit sets each step's time.
    t = path_timestamps([[0.0] * 7, [0.0] * 6 + [5.22]], vlim)
    assert abs(t[-1] - 2.0) < 1e-9
    # Leading padding values (expand_configs prefix) are ignored.
    assert abs(path_timestamps([[9.0] + [0.0] * 7, [-9.0, 2.175] + [0.0] * 6], vlim)[-1] - 1.0) < 1e-9

    carry = [0.0] * 7
    away = [1.0] * 7
    # carry -> away -> carry -> away -> carry: two actions.
    path = [carry, away, carry, carry, away, away, carry]
    assert split_at_carry(path, carry) == [(0, 2), (3, 6)], split_at_carry(path, carry)
    # Last action never returns to carry (e.g. the composite path just ends there).
    assert split_at_carry([carry, away, carry, away], carry) == [(0, 2), (2, 3)]
    # Idling at carry at the end isn't an action.
    assert split_at_carry([carry, away, carry, carry], carry) == [(0, 2)]
    # A path that brushes carry_conf mid-transit: split_at_carry sees 3 motions, the grasp/release
    # events still give the right 2 actions (grasp at index 3, release at index 6).
    mid = [carry, away, carry, away, away, carry, away, carry]
    assert len(split_at_carry(mid, carry)) == 3
    assert split_at_events(mid, carry, [3, 6]) == [(0, 5), (5, 7)], split_at_events(mid, carry, [3, 6])
    # Idling at carry between actions is skipped: the second action starts when it leaves carry.
    assert split_at_events([carry, away, carry, carry, carry, away, carry], carry, [1, 5]) == [(0, 2), (4, 6)]
    # Final action that never returns to carry.
    assert split_at_events([carry, away, carry, away], carry, [1, 3]) == [(0, 2), (2, 3)]
    # Asynchronous execution: robot 0 runs a (0..1s) then b from its re-timed start 3s; robot 1
    # idles at its first conf until c starts at 0.5s.
    tl = {
        'a': {'robot_index': 0, 'type': 'transit', 'duration': 1.0, 'motion_times': [0.0, 1.0],
              'path': [[0.0] * 7, [1.0] * 7], 'event_index': 1, 'obj': 'o0'},
        'b': {'robot_index': 0, 'type': 'transfer', 'duration': 2.0, 'motion_times': [0.0, 1.0, 2.0],
              'path': [[1.0] * 7, [2.0] * 7, [1.0] * 7], 'event_index': 1, 'obj': 'o0'},
        'c': {'robot_index': 1, 'type': 'transit', 'duration': 1.0, 'motion_times': [0.0, 1.0],
              'path': [[0.0] * 7, [3.0] * 7], 'event_index': 1, 'obj': 'o1'},
    }
    ex = AsyncExecution(tl, {'a': 0.0, 'b': 3.0, 'c': 0.5}, 2)
    assert abs(ex.makespan - 5.0) < 1e-9
    assert abs(ex.conf(0, 0.5)[0] - 0.5) < 1e-9          # halfway through a
    assert abs(ex.conf(0, 2.0)[0] - 1.0) < 1e-9          # waiting after a, before b
    assert abs(ex.conf(0, 4.0)[0] - 2.0) < 1e-9          # b's grasp-free midpoint
    assert abs(ex.conf(1, 0.2)[0] - 0.0) < 1e-9          # robot 1 not started yet
    assert ex.active(0, 2.0) == (None, None) and ex.active(0, 3.5)[0] == 'b'
    assert [(e.action, e.kind, round(e.time, 6)) for e in ex.events] == \
        [('a', 'grasp', 1.0), ('c', 'grasp', 1.5), ('b', 'release', 4.0)]
    assert lockstep_makespan({'x': {'end': 2.0}, 'y': {'end': 7.5}}) == 7.5
    print("All motion_timing self-checks passed.")
