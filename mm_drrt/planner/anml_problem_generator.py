"""
ANML Problem Generator for MM-dRRT.

Converts MM-dRRT environment states (from create_pddl_problem()) into ANML
domain and problem files, and maintains a bidirectional mapping between
PyBullet IDs and ANML object names.
"""

import os
import itertools
from mm_drrt.planner.anml_domain import build_mm_drrt_anml_domain


# PDDL predicate name → ANML fluent name (only fluents defined in the ANML domain)
_PDDL_TO_ANML_FLUENT = {
    'robot-free':          'robot_free',
    'holding':             'holding',
    'obj-clear':           'obj_clear',
    'surface-accessible':  'surface_accessible',
    'robot-can-reach':     'robot_can_reach',
    'obj-location':        'obj_location',
}

# PDDL type name → ANML type name
_PDDL_TO_ANML_TYPE = {
    'robot':       'Robot',
    'movable-obj': 'MovableObj',
    'fixed-obj':   'FixedObj',
}

# Fluents that carry an object value rather than a boolean
_OBJECT_FLUENTS = {'obj-location', 'obj_location'}

# Argument PDDL types per boolean fluent — used to emit explicit false initializations
# (UPF ANML requires every fluent instantiation to have an explicit initial value)
_BOOLEAN_FLUENT_ARG_TYPES = {
    'robot_free':         ['robot'],
    'holding':            ['robot', 'movable-obj'],
    'obj_clear':          ['movable-obj'],
    'surface_accessible': ['fixed-obj'],
    'robot_can_reach':    ['robot', 'fixed-obj'],
}


class ObjectMapper:
    """Bidirectional mapping between PyBullet objects and ANML names."""

    def __init__(self):
        self.pybullet_to_anml = {}
        self.anml_to_pybullet = {}

    def register(self, pybullet_obj, anml_name):
        self.pybullet_to_anml[pybullet_obj] = anml_name
        self.anml_to_pybullet[anml_name.lower()] = pybullet_obj

    def get_anml_name(self, pybullet_obj):
        return self.pybullet_to_anml.get(pybullet_obj)

    def get_pybullet_obj(self, anml_name):
        return self.anml_to_pybullet.get(anml_name.lower())


def generate_anml_files(env, output_dir, transit_duration=10, transfer_duration=10):
    """
    Generate a single combined ANML file (domain + problem) from the environment.

    UPF's ANMLReader expects domain and problem in one file.

    Args:
        env: MM-dRRT environment with create_pddl_problem() method
        output_dir: Directory to write the combined .anml file
        transit_duration: Fixed duration for transit (pick) actions
        transfer_duration: Fixed duration for transfer (place) actions

    Returns:
        (combined_path, mapper)
    """
    if not hasattr(env, 'create_pddl_problem'):
        raise NotImplementedError(
            f"Environment {type(env).__name__} must implement create_pddl_problem()"
        )

    objects, init_state, goal_state = env.create_pddl_problem()
    mapper = ObjectMapper()

    os.makedirs(output_dir, exist_ok=True)
    combined_path = os.path.join(output_dir, 'problem.anml')

    domain_content = build_mm_drrt_anml_domain(
        transit_duration=transit_duration,
        transfer_duration=transfer_duration,
    )
    problem_content = _build_problem_anml(objects, init_state, goal_state, mapper)

    with open(combined_path, 'w', encoding='utf-8') as f:
        f.write(domain_content)
        f.write('\n')
        f.write(problem_content)

    return combined_path, mapper


def _build_problem_anml(objects, init_state, goal_state, mapper):
    lines = []

    # Object declarations — UPF ANML requires the 'instance' keyword
    instances_by_type = {}
    for obj_type, obj_list in objects.items():
        anml_type = _PDDL_TO_ANML_TYPE.get(obj_type)
        if anml_type is None:
            raise ValueError(f"Unknown object type: {obj_type}")
        names = []
        for i, pybullet_obj in enumerate(obj_list):
            anml_name = f"{obj_type.replace('-', '_')}_{i}"
            mapper.register(pybullet_obj, anml_name)
            lines.append(f"instance {anml_type} {anml_name};")
            names.append(anml_name)
        instances_by_type[obj_type] = names

    lines.append('')

    # Collect (anml_fluent, args_tuple) pairs that are explicitly true in init
    true_init = set()
    for predicate_tuple in init_state:
        predicate_name = predicate_tuple[0]
        predicate_args = predicate_tuple[1:]
        anml_fluent = _PDDL_TO_ANML_FLUENT.get(predicate_name)
        if anml_fluent is None or predicate_name in _OBJECT_FLUENTS:
            continue
        anml_args = tuple(mapper.get_anml_name(arg) for arg in predicate_args)
        true_init.add((anml_fluent, anml_args))

    # Emit true initial values
    for predicate_tuple in init_state:
        predicate_name = predicate_tuple[0]
        predicate_args = predicate_tuple[1:]

        anml_fluent = _PDDL_TO_ANML_FLUENT.get(predicate_name)
        if anml_fluent is None:
            continue  # skip predicates not in the ANML domain

        anml_args = [mapper.get_anml_name(arg) for arg in predicate_args]
        if None in anml_args:
            raise ValueError(f"Unmapped object in predicate {predicate_tuple}")

        if predicate_name in _OBJECT_FLUENTS:
            func_args = ', '.join(anml_args[:-1])
            value = anml_args[-1]
            lines.append(f"[start] {anml_fluent}({func_args}) := {value};")
        else:
            func_args = ', '.join(anml_args)
            lines.append(f"[start] {anml_fluent}({func_args}) := true;")

    # Emit explicit false for every boolean fluent combination not set to true.
    # UPF ANML requires every fluent instantiation to have an explicit initial value.
    for anml_fluent, arg_types in _BOOLEAN_FLUENT_ARG_TYPES.items():
        type_instance_lists = [instances_by_type.get(t, []) for t in arg_types]
        for combo in itertools.product(*type_instance_lists):
            if (anml_fluent, combo) not in true_init:
                func_args = ', '.join(combo)
                lines.append(f"[start] {anml_fluent}({func_args}) := false;")

    lines.append('')

    # Goal — boolean goals use bare fluent name; object goals use ==
    for predicate_tuple in goal_state:
        predicate_name = predicate_tuple[0]
        predicate_args = predicate_tuple[1:]

        anml_fluent = _PDDL_TO_ANML_FLUENT.get(predicate_name)
        if anml_fluent is None:
            continue

        anml_args = [mapper.get_anml_name(arg) for arg in predicate_args]
        if None in anml_args:
            raise ValueError(f"Unmapped object in goal {predicate_tuple}")

        if predicate_name in _OBJECT_FLUENTS:
            func_args = ', '.join(anml_args[:-1])
            value = anml_args[-1]
            lines.append(f"[end] {anml_fluent}({func_args}) == {value};")
        else:
            func_args = ', '.join(anml_args)
            lines.append(f"[end] {anml_fluent}({func_args});")

    lines.append('')
    return '\n'.join(lines)
