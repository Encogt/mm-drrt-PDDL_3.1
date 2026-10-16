"""
PDDL Problem Generator for MM-dRRT

Converts MM-dRRT environment states into PDDL problem instances and maintains
bidirectional mapping between PyBullet IDs and PDDL names.

Supports PDDL 2.1 durative actions with boolean fluents (see pddl_domain.py).
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


def generate_problem(env, save_to_file=False, transit_duration=10, transfer_duration=10,
                     region_mutex_enabled=False,
                     transit_region_entry_offset=0, transit_region_exit_offset=None,
                     transfer_region_entry_offset=0, transfer_region_exit_offset=None,
                     use_region_fluents=False, region_offsets=None):
    """
    Create a PDDL 2.1 temporal problem instance from the environment.

    Args:
        env: MM-dRRT environment with create_pddl_problem() method
        save_to_file: If True, write UPF text representation to mm_drrt/pddl/problems/
        transit_duration: Fixed duration (seconds) for transit (pick) actions
        transfer_duration: Fixed duration (seconds) for transfer (place) actions
        region_mutex_enabled: If False (default), the generated domain has no region-conflict
            handling at all -- matches the domain's historic behavior exactly (see
            create_mm_drrt_domain's docstring; some environments, e.g.
            DurationConflictRaiEnvironment, depend on this). Must be True for the region_*_offset
            args below to have any effect.
        transit_region_entry_offset/transit_region_exit_offset: sub-interval of a transit action
            (of transit_duration) during which it actually occupies its shared region -- see
            create_mm_drrt_domain's docstring. Defaults reproduce a full-duration mutex. Under
            use_region_fluents=True these become the fallback for any fixed-obj region_offsets
            doesn't have a measurement for, rather than a single domain-wide constant.
        transfer_region_entry_offset/transfer_region_exit_offset: same, for transfer actions.
        use_region_fluents: If True, alpha_ik/beta_ik are per-(robot, fixed-obj) numeric fluents
            (see create_mm_drrt_domain's docstring) instead of domain-wide constants, and are set
            per (robot, fixed-obj) pair from `region_offsets` below (falling back to the
            transit_/transfer_ *_offset constants above for any pair not in it).
        region_offsets: {(pybullet_robot, pybullet_fixed_obj): {'transit': (alpha, beta) or None,
            'transfer': (alpha, beta) or None}} -- real, motion-derived per-(robot, region)
            offsets (see mm_drrt/utils/rai_motion_planner_utils.py's sample_region_offsets()),
            keyed by the SAME pybullet object identifiers env.create_pddl_problem() returns in its
            'robot'/'fixed-obj' lists -- one entry per action i (a specific robot's move), not
            just per region, since two robots can approach the same region with different
            geometry. Only consulted when use_region_fluents=True. A None value (sampling found no
            valid grasp/IK/motion for that pair) is treated the same as a missing entry -- the
            constant fallback above is used instead of silently leaving a region unprotected.

    Returns:
        (problem, mapper)
    """
    objects, init_state, goal_state = env.create_pddl_problem()

    mapper  = ObjectMapper()
    problem = Problem('mm-drrt-problem')

    domain          = create_mm_drrt_domain(transit_duration=transit_duration,
                                            transfer_duration=transfer_duration,
                                            region_mutex_enabled=region_mutex_enabled,
                                            transit_region_entry_offset=transit_region_entry_offset,
                                            transit_region_exit_offset=transit_region_exit_offset,
                                            transfer_region_entry_offset=transfer_region_entry_offset,
                                            transfer_region_exit_offset=transfer_region_exit_offset,
                                            use_region_fluents=use_region_fluents)
    types           = domain['types']
    boolean_fluents = domain['boolean_fluents']
    numeric_fluents = domain['numeric_fluents']
    actions         = domain['actions']

    for fluent in boolean_fluents:
        problem.add_fluent(fluent, default_initial_value=False)
    for fluent, default_value in numeric_fluents:
        problem.add_fluent(fluent, default_initial_value=default_value)
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

    # Per-region motion-derived overrides (only meaningful with use_region_fluents=True -- see
    # this function's docstring). A region with no measurement here keeps the domain-wide
    # constant default_initial_value already set above.
    if region_offsets:
        offset_fluents = domain['region_offset_fluents']
        for (pybullet_robot, pybullet_f_obj), per_type in region_offsets.items():
            robot_pddl_name = mapper.get_pddl_name(pybullet_robot)
            f_obj_pddl_name = mapper.get_pddl_name(pybullet_f_obj)
            if robot_pddl_name is None or f_obj_pddl_name is None:
                continue
            upf_robot = _lookup_upf_obj(robot_pddl_name, problem)
            upf_f_obj = _lookup_upf_obj(f_obj_pddl_name, problem)
            for action_type, fluent_keys in (('transit', ('transit_entry', 'transit_exit')),
                                             ('transfer', ('transfer_entry', 'transfer_exit'))):
                measured = (per_type or {}).get(action_type)
                if measured is None:
                    continue
                alpha, beta = measured
                problem.set_initial_value(offset_fluents[fluent_keys[0]](upf_robot, upf_f_obj), float(alpha))
                problem.set_initial_value(offset_fluents[fluent_keys[1]](upf_robot, upf_f_obj), float(beta))

    fluent_map = {f.name: f for f in boolean_fluents}

    # Set initial state
    for predicate_tuple in init_state:
        predicate_name = predicate_tuple[0]
        predicate_args = predicate_tuple[1:]

        if predicate_name not in fluent_map:
            raise ValueError(f"Unknown predicate: {predicate_name}")

        fluent   = fluent_map[predicate_name]
        upf_args = _resolve_args(predicate_args, mapper, problem)
        problem.set_initial_value(fluent(*upf_args), True)

    # Set goal
    goal_conditions = []
    for predicate_tuple in goal_state:
        predicate_name = predicate_tuple[0]
        predicate_args = predicate_tuple[1:]

        if predicate_name not in fluent_map:
            raise ValueError(f"Unknown predicate: {predicate_name}")

        fluent   = fluent_map[predicate_name]
        upf_args = _resolve_args(predicate_args, mapper, problem)
        goal_conditions.append(fluent(*upf_args))

    if len(goal_conditions) == 1:
        problem.add_goal(goal_conditions[0])
    else:
        problem.add_goal(And(*goal_conditions))

    if save_to_file:
        import os
        problem_dir = os.path.join(os.path.dirname(__file__), '../pddl/problems/')
        os.makedirs(problem_dir, exist_ok=True)
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
