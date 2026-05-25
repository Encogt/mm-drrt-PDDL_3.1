# PDDL 3.1 Integration for MM-dRRT

This fork extends the original [MM-dRRT](https://github.com/syc7446/mm-drrt) with PDDL 3.1 planner integration for automatic task plan generation.

## Overview

The PDDL integration allows MM-dRRT to automatically generate task plan skeletons using standard PDDL planners, rather than requiring manual plan specification. The PDDL planner generates the high-level action sequence, and MM-dRRT refines it into executable motion plans.

## PDDL 3.1 Feature: Object Fluents

The domain uses the `:object-fluents` requirement from PDDL 3.1. Instead of a boolean predicate `(obj-on ?m ?f)`, object location is represented by a function:

```pddl
(:functions
  (obj-location ?m - movable-obj) - fixed-obj
)
```

This means:
- **Precondition**: `(= (obj-location ?m) ?from)` — checks which surface the object is on
- **Effect**: `(assign (obj-location ?m) ?to)` — updates the surface after placement

This is more expressive than boolean predicates: a single fluent value captures where each object is, rather than requiring one predicate per possible (object, surface) pair.

## What Changed (PDDL 3.1 Upgrade)

The original implementation used `:strips :typing` — basic classical PDDL — and was incorrectly labelled PDDL 3.1. The following changes make it genuinely PDDL 3.1.

| File | Change |
|---|---|
| `mm_drrt/pddl/domains/mm_drrt_manipulation.pddl` | Added `:object-fluents` requirement; replaced `obj-on` boolean predicate with `(:functions (obj-location ?m) - fixed-obj)`; `transit` precondition now uses `(= (obj-location ?m) ?from)`; `transfer` effect now uses `(assign (obj-location ?m) ?to)` |
| `mm_drrt/planner/pddl_domain.py` | Object fluent created as `Fluent('obj-location', FixedObj, m=MovableObj)`; precondition uses `Equals(obj_location(m), from_f)`; place effect uses `transfer.add_effect(obj_location(m), to_f)`; return value now separates `boolean_fluents` and `object_fluents` |
| `mm_drrt/planner/pddl_problem_generator.py` | Detects fluent type via `fluent.type.is_bool_type()`; object fluents use `set_initial_value(fluent(key), value)` for init and `Equals(fluent(key), value)` for goal |
| `mm_drrt/planner/pddl_planner.py` | Default planner is auto-selected (`None`); uses `OneshotPlanner(problem_kind=problem.kind)` so a planner supporting `:object-fluents` (Tamer) is selected automatically |
| `examples/envs/example_single_robot_env.py` | `('obj-on', m, f)` → `('obj-location', m, f)` in both `init_state` and `goal_state` |
| `requirements.txt` | Added `up-tamer` (required for `:object-fluents` support) |

## Core Integration Components

1. **PDDL Domain** — `mm_drrt/pddl/domains/mm_drrt_manipulation.pddl`
   - Requirements: `:strips :typing :object-fluents`
   - Object fluent: `obj-location(m) → fixed-obj`
   - Boolean predicates: `robot-free`, `holding`, `obj-clear`, `surface-accessible`
   - Actions: `transit` (pick), `transfer` (place)

2. **PDDL Planner Module** — `mm_drrt/planner/pddl_planner.py`
   - Main orchestrator using Unified Planning Framework
   - Auto-selects a planner that supports PDDL 3.1 object fluents
   - Configurable timeout and error handling

3. **Problem Generator** — `mm_drrt/planner/pddl_problem_generator.py`
   - Converts MM-dRRT environment state to a UPF Problem instance
   - Handles both boolean fluents and object-typed fluents
   - Bidirectional object mapping (PyBullet ↔ PDDL)

4. **Plan Parser** — `mm_drrt/utils/pddl_parser.py`
   - Converts PDDL plans to MM-dRRT format
   - Extracts: `plan`, `action_orders`, `obj_orders`, `init_order_constraints`
   - Supports sequential and temporal plans

5. **Domain Builder** — `mm_drrt/planner/pddl_domain.py`
   - Programmatic domain creation using UPF API
   - Uses `Fluent('obj-location', FixedObj, m=MovableObj)` for the PDDL 3.1 object fluent
   - Uses `Equals(obj_location(m), from_f)` for preconditions

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/mm-drrt-pddl.git
cd mm-drrt-pddl
pip install -r requirements.txt
```

`requirements.txt` includes `up-tamer`, which is the planner backend used for PDDL 3.1 object fluent support.

## Usage

### With PDDL Planner (Automatic Plan Generation)

```bash
python main.py --use_pddl_planner --env_type exp_single_robot --num_robots 1 --num_objs 1
```

### Without PDDL Planner (Manual Plan Specification)

```bash
python main.py --env_type exp_single_robot --num_robots 1 --num_objs 1
```

## How It Works

```
1. Environment Specification
   ↓
   env.create_pddl_problem()
   → Returns: (objects, init_state, goal_state)

2. PDDL Problem Generation
   ↓
   generate_problem()
   → Creates UPF Problem with object fluents
   → ('obj-location', m, f)  →  set_initial_value(obj_location(m), f)
   → ('obj-location', m, f) in goals  →  Equals(obj_location(m), f)

3. Planning
   ↓
   OneshotPlanner(problem_kind=problem.kind)
   → Auto-selects Tamer (supports :object-fluents)
   → Returns a satisficing plan

4. Plan Parsing
   ↓
   parse_pddl_plan()
   → Converts to MM-dRRT format:
     plan: {'a0': ('transit', robot, obj, None, table_0), ...}
     action_orders: {robot_0: ('a0', 'a1')}
     obj_orders: {obj_0: ['a1']}
     init_order_constraints: (...)

5. MM-dRRT Refinement (Unchanged)
   ↓
   Placement → Subgoal → Path computation → dRRT*
```

## Architecture

```
mm_drrt/
├── planner/
│   ├── pddl_planner.py           Main PDDL planner orchestrator
│   ├── pddl_domain.py            Programmatic PDDL 3.1 domain builder
│   ├── pddl_problem_generator.py Environment → UPF Problem converter
│   └── task_planner.py           MM-dRRT plan refinement (existing)
├── utils/
│   └── pddl_parser.py            PDDL plan → MM-dRRT converter
└── pddl/
    └── domains/
        └── mm_drrt_manipulation.pddl  PDDL 3.1 domain definition
```

## Adding PDDL Support to a Custom Environment

```python
def create_pddl_problem(self):
    objects = {
        'robot': [self.robots[0]],
        'movable-obj': [self.m_objs[0]],
        'fixed-obj': [self.f_objs[0], self.f_objs[1]]
    }

    init_state = [
        ('robot-free', self.robots[0]),
        ('robot-at-base', self.robots[0]),
        ('obj-location', self.m_objs[0], self.f_objs[0]),  # object fluent assignment
        ('obj-clear', self.m_objs[0]),
        ('surface-accessible', self.f_objs[0]),
        ('surface-accessible', self.f_objs[1])
    ]

    goal_state = [
        ('obj-location', self.m_objs[0], self.f_objs[1]),  # object fluent equality goal
        ('robot-free', self.robots[0])
    ]

    return objects, init_state, goal_state
```

Tuples with an object-typed fluent name (e.g., `obj-location`) are handled automatically:
- In `init_state`: interpreted as `set_initial_value(fluent(key_args), value)`
- In `goal_state`: interpreted as `Equals(fluent(key_args), value)`

## Citation

```bibtex
@inproceedings{sung2024mmdrrt,
  title={Asynchronous Task Plan Refinement for Multi-Robot Task and Motion Planning},
  author={Sung, Yoonchang and Shome, Rahul and Stone, Peter},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2024},
  organization={IEEE}
}
```

## Acknowledgments

- Original MM-dRRT: [syc7446/mm-drrt](https://github.com/syc7446/mm-drrt)
- Unified Planning Framework: [aiplan4eu/unified-planning](https://github.com/aiplan4eu/unified-planning)
- Tamer: FBK Tamer Development Team
