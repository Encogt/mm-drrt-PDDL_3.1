"""
PDDL Problem Generator for MM-dRRT

This module converts MM-dRRT environment states into PDDL problem instances
and maintains bidirectional mapping between PyBullet IDs and PDDL names.
"""

from unified_planning.shortcuts import *
from unified_planning.io import PDDLWriter
from mm_drrt.planner.pddl_domain import create_mm_drrt_domain


class ObjectMapper:
    """
    Maintains bidirectional mapping between PyBullet objects and PDDL names.
    """

    def __init__(self):
        self.pybullet_to_pddl = {}  # {pybullet_id: 'pddl_name'}
        self.pddl_to_pybullet = {}  # {'pddl_name': pybullet_id}

    def register(self, pybullet_obj, pddl_name):
        """Register a PyBullet object with its PDDL name."""
        self.pybullet_to_pddl[pybullet_obj] = pddl_name
        self.pddl_to_pybullet[pddl_name] = pybullet_obj

    def get_pddl_name(self, pybullet_obj):
        """Get PDDL name for a PyBullet object."""
        return self.pybullet_to_pddl.get(pybullet_obj)

    def get_pybullet_obj(self, pddl_name):
        """Get PyBullet object for a PDDL name."""
        return self.pddl_to_pybullet.get(pddl_name)


def generate_problem(env, save_to_file=False):
    """
    Create PDDL problem instance from environment.

    Args:
        env: MM-dRRT environment instance with create_pddl_problem() method
        save_to_file: If True, saves problem to file in mm_drrt/pddl/problems/

    Returns:
        tuple: (problem, mapper)
            problem: UPF Problem instance
            mapper: ObjectMapper for converting between PDDL and PyBullet IDs
    """
    # Get problem specification from environment
    objects, init_state, goal_state = env.create_pddl_problem()

    # Create object mapper
    mapper = ObjectMapper()

    # Create problem
    problem = Problem('mm-drrt-problem')

    # Get domain components
    domain_components = create_mm_drrt_domain()
    types = domain_components['types']
    fluents = domain_components['fluents']
    actions = domain_components['actions']

    # Add fluents to problem
    for fluent in fluents:
        problem.add_fluent(fluent, default_initial_value=False)

    # Add actions to problem
    for action in actions:
        problem.add_action(action)

    # Create UPF objects and register mappings
    upf_objects = {}

    for obj_type, obj_list in objects.items():
        if obj_type not in types:
            raise ValueError(f"Unknown object type: {obj_type}")

        upf_objects[obj_type] = []
        for i, pybullet_obj in enumerate(obj_list):
            # Create PDDL name
            pddl_name = f"{obj_type.replace('-', '_')}_{i}"

            # Create UPF object
            upf_obj = Object(pddl_name, types[obj_type])
            problem.add_object(upf_obj)

            # Register mapping
            mapper.register(pybullet_obj, pddl_name)
            upf_objects[obj_type].append(upf_obj)

    # Set initial state
    fluent_map = {f.name: f for f in fluents}

    for predicate_tuple in init_state:
        predicate_name = predicate_tuple[0]
        predicate_args = predicate_tuple[1:]

        if predicate_name not in fluent_map:
            raise ValueError(f"Unknown predicate: {predicate_name}")

        fluent = fluent_map[predicate_name]

        # Convert PyBullet objects to UPF objects
        upf_args = []
        for arg in predicate_args:
            pddl_name = mapper.get_pddl_name(arg)
            if pddl_name is None:
                raise ValueError(f"Object not registered: {arg}")

            # Find UPF object by name
            upf_obj = None
            for obj in problem.all_objects:
                if obj.name == pddl_name:
                    upf_obj = obj
                    break

            if upf_obj is None:
                raise ValueError(f"UPF object not found for: {pddl_name}")

            upf_args.append(upf_obj)

        # Set initial value
        problem.set_initial_value(fluent(*upf_args), True)

    # Set goal state
    goal_conditions = []

    for predicate_tuple in goal_state:
        predicate_name = predicate_tuple[0]
        predicate_args = predicate_tuple[1:]

        if predicate_name not in fluent_map:
            raise ValueError(f"Unknown predicate: {predicate_name}")

        fluent = fluent_map[predicate_name]

        # Convert PyBullet objects to UPF objects
        upf_args = []
        for arg in predicate_args:
            pddl_name = mapper.get_pddl_name(arg)
            if pddl_name is None:
                raise ValueError(f"Object not registered: {arg}")

            # Find UPF object by name
            upf_obj = None
            for obj in problem.all_objects:
                if obj.name == pddl_name:
                    upf_obj = obj
                    break

            if upf_obj is None:
                raise ValueError(f"UPF object not found for: {pddl_name}")

            upf_args.append(upf_obj)

        goal_conditions.append(fluent(*upf_args))

    # Set goal
    if len(goal_conditions) == 1:
        problem.add_goal(goal_conditions[0])
    else:
        problem.add_goal(And(*goal_conditions))

    # Optionally save to file
    if save_to_file:
        import os
        problem_dir = os.path.join(
            os.path.dirname(__file__),
            '../pddl/problems/'
        )
        os.makedirs(problem_dir, exist_ok=True)

        writer = PDDLWriter(problem)
        writer.write_domain(os.path.join(problem_dir, 'domain.pddl'))
        writer.write_problem(os.path.join(problem_dir, 'problem.pddl'))

    return problem, mapper


def extract_objects_from_env(env):
    """
    Extract typed objects from environment (helper method).

    Args:
        env: MM-dRRT environment instance

    Returns:
        dict: Object type → list of objects
    """
    # This is a helper - actual extraction happens in env.create_pddl_problem()
    if not hasattr(env, 'create_pddl_problem'):
        raise NotImplementedError(
            f"Environment {type(env).__name__} must implement create_pddl_problem()"
        )

    objects, _, _ = env.create_pddl_problem()
    return objects


def generate_init_state(env):
    """
    Generate initial state predicates (helper method).

    Args:
        env: MM-dRRT environment instance

    Returns:
        list: Initial state predicates
    """
    if not hasattr(env, 'create_pddl_problem'):
        raise NotImplementedError(
            f"Environment {type(env).__name__} must implement create_pddl_problem()"
        )

    _, init_state, _ = env.create_pddl_problem()
    return init_state


def generate_goal_state(env):
    """
    Generate goal state predicates (helper method).

    Args:
        env: MM-dRRT environment instance

    Returns:
        list: Goal state predicates
    """
    if not hasattr(env, 'create_pddl_problem'):
        raise NotImplementedError(
            f"Environment {type(env).__name__} must implement create_pddl_problem()"
        )

    _, _, goal_state = env.create_pddl_problem()
    return goal_state
