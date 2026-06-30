"""
Tamer Plan Parser for MM-dRRT.

Converts a temporal plan produced by Tamer (from an ANML problem) into
MM-dRRT's expected format: (plan, action_orders, obj_orders, init_order_constraints).

Tamer outputs temporal plans in one of these formats:
  0.0: transit(robot_0, movable_obj_0, fixed_obj_0) [10.0]
  0.0: [transit robot_0 movable_obj_0 fixed_obj_0] [10.0]
"""

import re
from collections import defaultdict


def parse_anml_upf_plan(upf_plan, mapper, env):
    """
    Convert a UPF TimeTriggeredPlan (returned by Tamer after solving an ANML
    problem) to MM-dRRT format.

    Args:
        upf_plan: TimeTriggeredPlan from UPF/Tamer
        mapper:   ObjectMapper from anml_problem_generator (anml_name → pybullet_id)
        env:      Environment instance

    Returns:
        (plan, action_orders, obj_orders, init_order_constraints)
    """
    plan = {}
    actions_list = []

    for i, (start_time, action, duration) in enumerate(
            sorted(upf_plan.timed_actions, key=lambda t: t[0])):
        action_name = f"a{i}"
        action_type = action.action.name
        args = [_upf_param_name(p) for p in action.actual_parameters]
        action_info = _build_action_info(action_type, args, action_name, mapper,
                                         float(start_time),
                                         float(start_time) + float(duration))
        if action_info is None:
            continue
        actions_list.append(action_info)
        plan[action_name] = action_info['mm_drrt_format']

    action_orders = defaultdict(list)
    for action_info in actions_list:
        robot = action_info['robot']
        action_orders[robot].append((action_info['start_time'], action_info['name']))
    for robot in action_orders:
        sorted_acts = sorted(action_orders[robot], key=lambda x: x[0])
        action_orders[robot] = tuple(a[1] for a in sorted_acts)

    obj_orders = defaultdict(list)
    for action_info in actions_list:
        if action_info['type'] == 'transfer':
            obj_orders[action_info['movable_obj']].append(action_info['name'])

    init_order_constraints = _extract_temporal_constraints(actions_list)

    return plan, dict(action_orders), dict(obj_orders), init_order_constraints


def _upf_param_name(parameter):
    if hasattr(parameter, 'object'):
        return parameter.object().name
    if hasattr(parameter, 'name'):
        return parameter.name
    return str(parameter)


def parse_tamer_plan(plan_lines, mapper, env):
    """
    Convert a Tamer temporal plan to MM-dRRT format.

    Args:
        plan_lines: List of plan output lines from Tamer
        mapper: ObjectMapper (anml_name → pybullet_id)
        env: Environment instance

    Returns:
        (plan, action_orders, obj_orders, init_order_constraints)
    """
    plan = {}
    actions_list = []

    for raw_line in plan_lines:
        action_info = _parse_tamer_line(raw_line, f"a{len(actions_list)}", mapper)
        if action_info is None:
            continue
        actions_list.append(action_info)
        plan[action_info['name']] = action_info['mm_drrt_format']

    action_orders = defaultdict(list)
    for action_info in actions_list:
        robot = action_info['robot']
        action_orders[robot].append((action_info['start_time'], action_info['name']))

    for robot in action_orders:
        sorted_acts = sorted(action_orders[robot], key=lambda x: x[0])
        action_orders[robot] = tuple(a[1] for a in sorted_acts)

    obj_orders = defaultdict(list)
    for action_info in actions_list:
        if action_info['type'] == 'transfer':
            obj_orders[action_info['movable_obj']].append(action_info['name'])

    init_order_constraints = _extract_temporal_constraints(actions_list)

    return plan, dict(action_orders), dict(obj_orders), init_order_constraints


# ---------------------------------------------------------------------------
# Line parsing
# ---------------------------------------------------------------------------

# Matches: "0.0: transit(robot_0, movable_obj_0, fixed_obj_0) [10.0]"
_RE_FUNC = re.compile(
    r'^\s*(?P<t>[0-9]+(?:\.[0-9]*)?)[\s:]+(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)'
    r'\s*\((?P<args>[^)]*)\)'
    r'(?:\s*\[(?P<dur>[0-9.]+)\])?'
)

# Matches: "0.0: [transit robot_0 movable_obj_0 fixed_obj_0] [10.0]"
_RE_BRACKET = re.compile(
    r'^\s*(?P<t>[0-9]+(?:\.[0-9]*)?)[\s:]+\[(?P<body>[^\]]+)\]'
    r'(?:\s*\[(?P<dur>[0-9.]+)\])?'
)

# Matches ANML-style: "[0, 10] transit(robot_0, movable_obj_0, fixed_obj_0);"
_RE_ANML = re.compile(
    r'^\s*\[\s*(?P<s>[0-9.]+)\s*,\s*(?P<e>[0-9.]+)\s*\]\s*'
    r'(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*\((?P<args>[^)]*)\)'
)


def _parse_tamer_line(raw_line, action_name, mapper):
    line = raw_line.strip()
    if not line or line.startswith(';') or line.startswith('//'):
        return None

    # Try functional notation: "t: name(args) [dur]"
    m = _RE_FUNC.match(line)
    if m:
        start_time = float(m.group('t'))
        dur = float(m.group('dur')) if m.group('dur') else 10.0
        name = m.group('name').lower()
        args = [a.strip().lower() for a in m.group('args').split(',') if a.strip()]
        return _build_action_info(name, args, action_name, mapper, start_time, start_time + dur)

    # Try bracket notation: "t: [name arg1 arg2 ...] [dur]"
    m = _RE_BRACKET.match(line)
    if m:
        start_time = float(m.group('t'))
        dur = float(m.group('dur')) if m.group('dur') else 10.0
        tokens = m.group('body').lower().split()
        if not tokens:
            return None
        name = tokens[0]
        args = tokens[1:]
        return _build_action_info(name, args, action_name, mapper, start_time, start_time + dur)

    # Try ANML interval notation: "[s, e] name(args);"
    m = _RE_ANML.match(line)
    if m:
        start_time = float(m.group('s'))
        end_time = float(m.group('e'))
        name = m.group('name').lower()
        args = [a.strip().lower() for a in m.group('args').split(',') if a.strip()]
        return _build_action_info(name, args, action_name, mapper, start_time, end_time)

    return None


def _build_action_info(action_type, args, action_name, mapper, start_time, end_time):
    if not args:
        return None

    robot_pddl = args[0]
    robot = mapper.get_pybullet_obj(robot_pddl)

    if action_type == 'transit':
        if len(args) < 3:
            return None
        movable_obj = mapper.get_pybullet_obj(args[1])
        from_fixed_obj = mapper.get_pybullet_obj(args[2])
        mm_drrt_format = ('transit', robot, movable_obj, None, from_fixed_obj)
        return {
            'name': action_name,
            'type': 'transit',
            'robot': robot,
            'movable_obj': movable_obj,
            'from_fixed_obj': from_fixed_obj,
            'to_fixed_obj': from_fixed_obj,
            'mm_drrt_format': mm_drrt_format,
            'start_time': start_time,
            'end_time': end_time,
        }

    elif action_type == 'transfer':
        if len(args) < 3:
            return None
        movable_obj = mapper.get_pybullet_obj(args[1])
        to_fixed_obj = mapper.get_pybullet_obj(args[2])
        mm_drrt_format = ('transfer', robot, movable_obj, None, to_fixed_obj)
        return {
            'name': action_name,
            'type': 'transfer',
            'robot': robot,
            'movable_obj': movable_obj,
            'from_fixed_obj': None,
            'to_fixed_obj': to_fixed_obj,
            'mm_drrt_format': mm_drrt_format,
            'start_time': start_time,
            'end_time': end_time,
        }

    return None


def _extract_temporal_constraints(actions_list):
    constraints = []
    sorted_actions = sorted(actions_list, key=lambda a: a['start_time'])
    for i, a_i in enumerate(sorted_actions):
        if a_i['type'] != 'transfer':
            continue
        for a_j in sorted_actions[i + 1:]:
            if a_j['robot'] == a_i['robot']:
                continue
            if a_j['type'] == 'transit' and a_j['movable_obj'] == a_i['movable_obj']:
                constraints.append({'pre': a_i['name'], 'post': a_j['name']})
                break
    return tuple(constraints)
