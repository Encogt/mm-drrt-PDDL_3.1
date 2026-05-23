"""
PDDL Problem Generator for MM-dRRT

Converts MM-dRRT environment states into PDDL problem instances and maintains
bidirectional mapping between PyBullet IDs and PDDL names.

Supports PDDL 3.1 object fluents (obj-location) combined with durative actions.
"""

from unified_planning.shortcuts import *
from unified_planning.io import PDDLWriter
from mm_drrt.planner.pddl_domain import create_mm_drrt_domain


class ObjectMapper:
    """Bidirectional mapping between PyBullet objects and PDDL names."""

    def __init__(self):
        self.pybullet_to_pddl = {}
        self.pddl_to_pybullet = {}

    def register(self, pybullet_obj, pddl_name):
        self.pybullet_to_pddl[pybullet_obj] = pddl_name
        self.pddl_to_pybullet[pddl_name] = pybullet_obj

    def get_pddl_name(self, pybullet_obj):
        return self.pybullet_to_pddl.get(pybullet_obj)

    def get_pybullet_obj(self, pddl_name):
        return self.pddl_to_pybullet.get(pddl_name)


def _lookup_upf_obj(name, problem):
    for obj in problem.all_objects:
        if obj.name == name:
            return obj
    raise ValueError(f"UPF object not found for: {name}")


def _resolve_args(predicate_args, mapper, problem):
    upf_args = []
    for arg in predicate_args:
        pddl_name = mapper.get_pddl_name(arg)
        if pddl_name is None:
            raise ValueError(f"Object not registered in mapper: {arg}")
        upf_args.append(_lookup_upf_obj(pddl_name, problem))
    return upf_args


def generate_problem(env, save_to_file=False, transit_duration=10, transfer_duration=10):
    """
    Create a PDDL 3.1 temporal problem instance from the environment.

    Args:
        env: MM-dRRT environment with create_pddl_problem() method
        save_to_file: If True, write UPF text representation to mm_drrt/pddl/problems/
        transit_duration: Fixed duration (seconds) for transit (pick) actions
        transfer_duration: Fixed duration (seconds) for transfer (place) actions

    Returns:
        (problem, mapper)
    """
    objects, init_state, goal_state = env.create_pddl_problem()

    mapper  = ObjectMapper()
    problem = Problem('mm-drrt-problem')

    domain          = create_mm_drrt_domain(transit_duration=transit_duration,
                                            transfer_duration=transfer_duration)
    types           = domain['types']
    boolean_fluents = domain['boolean_fluents']
    object_fluents  = domain['object_fluents']
    actions         = domain['actions']

    for fluent in boolean_fluents:
        problem.add_fluent(fluent, default_initial_value=False)
    for fluent in object_fluents:
        problem.add_fluent(fluent)
    for action in actions:
        problem.add_action(action)

    # Build UPF objects and register mappings
    for obj_type, obj_list in objects.items():
        if obj_type not in types:
            raise ValueError(f"Unknown object type: {obj_type}")
        for i, pybullet_obj in enumerate(obj_list):
            pddl_name = f"{obj_type.replace('-', '_')}_{i}"
            upf_obj   = Object(pddl_name, types[obj_type])
            problem.add_object(upf_obj)
            mapper.register(pybullet_obj, pddl_name)

    all_fluents = boolean_fluents + object_fluents
    fluent_map  = {f.name: f for f in all_fluents}

    # Set initial state
    for predicate_tuple in init_state:
        predicate_name = predicate_tuple[0]
        predicate_args = predicate_tuple[1:]

        if predicate_name not in fluent_map:
            raise ValueError(f"Unknown predicate: {predicate_name}")

        fluent   = fluent_map[predicate_name]
        upf_args = _resolve_args(predicate_args, mapper, problem)

        if fluent.type.is_bool_type():
            problem.set_initial_value(fluent(*upf_args), True)
        else:
            # Object fluent: ('obj-location', m, f) → obj-location(m) = f
            problem.set_initial_value(fluent(*upf_args[:-1]), upf_args[-1])

    # Set goal
    goal_conditions = []
    for predicate_tuple in goal_state:
        predicate_name = predicate_tuple[0]
        predicate_args = predicate_tuple[1:]

        if predicate_name not in fluent_map:
            raise ValueError(f"Unknown predicate: {predicate_name}")

        fluent   = fluent_map[predicate_name]
        upf_args = _resolve_args(predicate_args, mapper, problem)

        if fluent.type.is_bool_type():
            goal_conditions.append(fluent(*upf_args))
        else:
            goal_conditions.append(Equals(fluent(*upf_args[:-1]), upf_args[-1]))

    if len(goal_conditions) == 1:
        problem.add_goal(goal_conditions[0])
    else:
        problem.add_goal(And(*goal_conditions))

    if save_to_file:
        import os
        problem_dir = os.path.join(os.path.dirname(__file__), '../pddl/problems/')
        os.makedirs(problem_dir, exist_ok=True)
        # PDDLWriter does not support object fluents; write UPF's own representation
        with open(os.path.join(problem_dir, 'problem.txt'), 'w') as f:
            f.write(str(problem))

    return problem, mapper


def extract_objects_from_env(env):
    if not hasattr(env, 'create_pddl_problem'):
        raise NotImplementedError(
            f"Environment {type(env).__name__} must implement create_pddl_problem()"
        )
    objects, _, _ = env.create_pddl_problem()
    return objects


def generate_init_state(env):
    if not hasattr(env, 'create_pddl_problem'):
        raise NotImplementedError(
            f"Environment {type(env).__name__} must implement create_pddl_problem()"
        )
    _, init_state, _ = env.create_pddl_problem()
    return init_state


def generate_goal_state(env):
    if not hasattr(env, 'create_pddl_problem'):
        raise NotImplementedError(
            f"Environment {type(env).__name__} must implement create_pddl_problem()"
        )
    _, _, goal_state = env.create_pddl_problem()
    return goal_state
