"""
MA-PDDL generation, universal-pddl-parser serialization, and Fast Downward
integration for MM-dRRT.

Workflow:
  1. generate MA-PDDL domain/problem (CN notation, :multi-agent :concurrency-network)
  2. run serialize_cn/serialize.bin  →  classical PDDL domain + problem
  3. run Fast Downward on the compiled files
  4. return plan lines for pddl_parser.parse_pddl_plan()
"""

import os
import shutil
import subprocess
import sys
from collections import defaultdict


class ObjectMapper:
    """Bidirectional mapping between PyBullet objects and PDDL names."""

    def __init__(self):
        self.pybullet_to_pddl = {}
        self.pddl_to_pybullet = {}

    def register(self, pybullet_obj, pddl_name):
        self.pybullet_to_pddl[pybullet_obj] = pddl_name
        self.pddl_to_pybullet[pddl_name.lower()] = pybullet_obj

    def get_pddl_name(self, pybullet_obj):
        return self.pybullet_to_pddl.get(pybullet_obj)

    def get_pybullet_obj(self, pddl_name):
        return self.pddl_to_pybullet.get(pddl_name.lower())


def _format_typed_objects(objects_by_type):
    lines = []
    for object_type, object_names in objects_by_type.items():
        if not object_names:
            continue
        lines.append("    " + " ".join(object_names) + f" - {object_type}")
    return "\n".join(lines)


def _translate_fact(fact, mapper):
    predicate_name = fact[0]
    mapped_args = []
    for arg in fact[1:]:
        pddl_name = mapper.get_pddl_name(arg)
        if pddl_name is None:
            raise ValueError(f"Object not registered in mapper: {arg}")
        mapped_args.append(pddl_name)

    if predicate_name == 'obj-location':
        if len(mapped_args) != 2:
            raise ValueError(f"Invalid obj-location fact: {fact}")
        return ('at', mapped_args[0], mapped_args[1])

    return tuple([predicate_name] + mapped_args)


def _format_fact(fact):
    if len(fact) == 1:
        return f"    ({fact[0]})"
    return "    (" + " ".join(fact) + ")"


# ---------------------------------------------------------------------------
# MA-PDDL (CN notation) builders
# ---------------------------------------------------------------------------

def build_ma_domain_pddl(domain_name='mm-drrt-ma'):
    return f"""(define (domain {domain_name})
  (:requirements :typing :multi-agent :concurrency-network)

  (:types
    agent
    movable-obj
    fixed-obj
  )

  (:predicates
    (robot-free ?r - agent)
    (holding ?r - agent ?m - movable-obj)
    (obj-clear ?m - movable-obj)
    (at ?m - movable-obj ?f - fixed-obj)
    (surface-accessible ?f - fixed-obj)
    (robot-can-reach ?r - agent ?f - fixed-obj)
  )

  (:action transit
    :agent ?r - agent
    :parameters (?m - movable-obj ?from - fixed-obj)
    :precondition (and
      (robot-free ?r)
      (at ?m ?from)
      (obj-clear ?m)
      (surface-accessible ?from)
      (robot-can-reach ?r ?from)
    )
    :effect (and
      (not (robot-free ?r))
      (not (obj-clear ?m))
      (not (at ?m ?from))
      (holding ?r ?m)
    )
  )

  (:action transfer
    :agent ?r - agent
    :parameters (?m - movable-obj ?to - fixed-obj)
    :precondition (and
      (holding ?r ?m)
      (surface-accessible ?to)
      (robot-can-reach ?r ?to)
    )
    :effect (and
      (robot-free ?r)
      (not (holding ?r ?m))
      (obj-clear ?m)
      (at ?m ?to)
    )
  )

  ; At most one agent may operate on the same movable object at a time.
  (:concurrency-constraint obj-mutex
    :parameters (?m - movable-obj)
    :bounds (1 1)
    :actions ((transit 1) (transfer 1))
  )

)
"""


_TYPE_MAP = {'robot': 'agent'}

# Predicates defined in the MA domain; any fact not in this set is dropped
# to avoid crashing the universal parser on unknown predicates.
_MA_DOMAIN_PREDICATES = {
    'robot-free', 'holding', 'obj-clear', 'at', 'surface-accessible', 'robot-can-reach',
}


def build_ma_problem_pddl(objects, init_state, goal_state, mapper,
                           problem_name='mm-drrt-problem',
                           domain_name='mm-drrt-ma'):
    objects_by_type = defaultdict(list)
    for object_type, object_list in objects.items():
        pddl_type = _TYPE_MAP.get(object_type, object_type)
        for index, pybullet_obj in enumerate(object_list):
            pddl_name = f"{object_type.replace('-', '_')}_{index}"
            mapper.register(pybullet_obj, pddl_name)
            objects_by_type[pddl_type].append(pddl_name)

    raw_init = [_translate_fact(fact, mapper) for fact in init_state]
    raw_goal = [_translate_fact(fact, mapper) for fact in goal_state]
    init_facts = [f for f in raw_init if f[0] in _MA_DOMAIN_PREDICATES]
    goal_facts = [f for f in raw_goal if f[0] in _MA_DOMAIN_PREDICATES]

    objects_block = _format_typed_objects(objects_by_type)
    init_block = "\n".join(_format_fact(fact) for fact in init_facts)
    goal_block = "\n".join(_format_fact(fact) for fact in goal_facts)

    return f"""(define (problem {problem_name})
  (:domain {domain_name})

  (:objects
{objects_block}
  )

  (:init
{init_block}
  )

  (:goal
    (and
{goal_block}
    )
  )
)
"""


# ---------------------------------------------------------------------------
# File generation
# ---------------------------------------------------------------------------

def generate_problem_files(env, output_dir):
    """Generate MA-PDDL files from env, compile to classical PDDL, return paths."""
    if not hasattr(env, 'create_pddl_problem'):
        raise NotImplementedError(
            f"Environment {type(env).__name__} must implement create_pddl_problem()"
        )

    objects, init_state, goal_state = env.create_pddl_problem()
    mapper = ObjectMapper()

    os.makedirs(output_dir, exist_ok=True)
    ma_domain_path = os.path.join(output_dir, 'ma_domain.pddl')
    ma_problem_path = os.path.join(output_dir, 'ma_problem.pddl')
    cl_domain_path = os.path.join(output_dir, 'domain.pddl')
    cl_problem_path = os.path.join(output_dir, 'problem.pddl')

    with open(ma_domain_path, 'w', encoding='utf-8') as f:
        f.write(build_ma_domain_pddl())

    with open(ma_problem_path, 'w', encoding='utf-8') as f:
        f.write(build_ma_problem_pddl(objects, init_state, goal_state, mapper))

    serializer = find_serializer_command()
    if serializer:
        run_serializer(serializer, ma_domain_path, ma_problem_path,
                       cl_domain_path, cl_problem_path)
    else:
        raise FileNotFoundError(
            'universal-pddl-parser serialize_cn binary not found. '
            'Set UPDDL_SERIALIZER_CMD or build ~/universal-pddl-parser-multiagent.'
        )

    return cl_domain_path, cl_problem_path, mapper


# kept for backward compatibility (pddl_planner.py imports this name)
generate_classical_problem_files = generate_problem_files


# ---------------------------------------------------------------------------
# Universal PDDL parser (serialize_cn)
# ---------------------------------------------------------------------------

def find_serializer_command():
    env_cmd = os.environ.get('UPDDL_SERIALIZER_CMD')
    if env_cmd and os.path.exists(env_cmd):
        return env_cmd

    home = os.path.expanduser('~')
    candidates = [
        os.path.join(home, 'universal-pddl-parser-multiagent',
                     'examples', 'serialize_cn', 'serialize.bin'),
        '/opt/universal-pddl-parser-multiagent/examples/serialize_cn/serialize.bin',
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def run_serializer(serializer_cmd, ma_domain, ma_problem, cl_domain, cl_problem):
    """Compile MA-PDDL to classical PDDL using serialize_cn."""
    with open(cl_domain, 'w', encoding='utf-8') as dom_f, \
         open(cl_problem, 'w', encoding='utf-8') as prob_f:
        result = subprocess.run(
            [serializer_cmd, ma_domain, ma_problem],
            stdout=dom_f,
            stderr=prob_f,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f'serialize_cn failed (exit {result.returncode}). '
            f'Check {cl_domain} and {cl_problem} for details.'
        )


# ---------------------------------------------------------------------------
# Fast Downward
# ---------------------------------------------------------------------------

def find_fast_downward_command(preferred_command=None):
    candidates = []
    if preferred_command:
        candidates.append(preferred_command)
    env_command = os.environ.get('FAST_DOWNWARD_CMD')
    if env_command:
        candidates.append(env_command)

    home = os.path.expanduser('~')
    candidates.extend([
        os.path.join(home, 'fast-downward', 'fast-downward.py'),
        os.path.join(home, 'fast_downward', 'fast-downward.py'),
        '/opt/fast-downward/fast-downward.py',
        '/usr/local/fast-downward/fast-downward.py',
    ])
    candidates.extend(['fast-downward.py', 'fast-downward'])

    for candidate in candidates:
        if os.path.isabs(candidate) or os.path.sep in candidate:
            if os.path.exists(candidate):
                return candidate
        else:
            resolved = shutil.which(candidate)
            if resolved:
                return resolved

    return None


def run_fast_downward(domain_path, problem_path, plan_path, timeout_seconds,
                      search='astar(blind())', preferred_command=None):
    planner_command = find_fast_downward_command(preferred_command)
    if planner_command is None:
        raise FileNotFoundError(
            'Fast Downward was not found. Set FAST_DOWNWARD_CMD or install fast-downward.py.'
        )

    command = []
    if planner_command.endswith('.py'):
        command.extend([sys.executable, planner_command])
    else:
        command.append(planner_command)

    # Driver options must precede the domain/problem files.
    command.extend([
        '--overall-time-limit', f'{timeout_seconds}s',
        '--plan-file', plan_path,
        domain_path,
        problem_path,
        '--search', search,
    ])

    return subprocess.run(command, capture_output=True, text=True, check=False)


def read_plan_file(plan_path):
    if not os.path.exists(plan_path):
        return []

    plan_lines = []
    with open(plan_path, 'r', encoding='utf-8') as plan_file:
        for raw_line in plan_file:
            line = raw_line.strip()
            if not line or line.startswith(';'):
                continue
            plan_lines.append(line)
    return plan_lines
