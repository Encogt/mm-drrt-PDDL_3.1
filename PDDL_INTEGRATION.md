# PDDL Integration for MM-dRRT

This fork extends the original [MM-dRRT](https://github.com/syc7446/mm-drrt) with PDDL planner integration for automatic task plan generation.

## Overview

The PDDL integration allows MM-dRRT to automatically generate task plan skeletons using standard PDDL planners, rather than requiring manual plan specification. The PDDL planner generates the high-level action sequence, and MM-dRRT refines it into executable motion plans.

`--use_pddl_planner` tries two planners in order, and prints which one produced the plan:

1. **Tamer** (`mm_drrt/planner/tamer_pddl_planner.py`) — solves the domain directly as a Unified Planning Framework `Problem`, no compilation step. Tried first.
2. **Fast Downward** (`mm_drrt/planner/pddl_planner.py` + `classical_pddl.py`) — compiles an MA-PDDL domain down to classical PDDL via the `universal-pddl-parser-multiagent` serializer and calls Fast Downward. Used only if Tamer fails.
3. If both fail, MM-dRRT falls back to the environment's manual `create_plan_order_constraints()`.

## PDDL Level: 2.1, Not 3.1

The domain is PDDL 2.1: it uses durative actions (a PDDL 2.1 feature, for scheduling multiple robots' actions with overlapping durations) but represents object location as a plain boolean predicate:

```pddl
(:predicates
  (obj-location ?m - movable-obj ?f - fixed-obj)
)
```

- **Precondition**: `(obj-location ?m ?from)` — true while the object rests on that surface
- **Effect**: `transit` retracts `(obj-location ?m ?from)` when picking up; `transfer` asserts `(obj-location ?m ?to)` when placing down

This intentionally avoids PDDL 3.1's `:object-fluents` requirement (a function `obj-location(m) -> fixed-obj` returning the surface directly, updated via `(assign ...)`). An earlier version of this integration used object fluents; it was reverted to keep both planner paths — Tamer and Fast Downward — on the same PDDL 2.1 feature set, since `classical_pddl.py`'s MA-PDDL compilation only ever supported boolean predicates.

## Core Integration Components

1. **PDDL Domain** — `mm_drrt/pddl/domains/mm_drrt_manipulation.pddl`
   - Requirements: `:strips :typing :durative-actions :duration-inequalities`
   - Boolean predicates: `robot-free`, `holding`, `obj-clear`, `surface-accessible`, `robot-can-reach`, `obj-location`
   - Actions: `transit` (pick), `transfer` (place)
   - This file is a human-readable reference copy; `pddl_domain.py` builds the same domain programmatically for Tamer, and `classical_pddl.py` builds its own MA-PDDL domain for Fast Downward.

2. **Tamer Planner Module** — `mm_drrt/planner/tamer_pddl_planner.py`
   - `TamerPDDLPlanner`, tried first by `main.py`
   - Solves the UPF `Problem` directly via `OneshotPlanner(name='tamer')`
   - Configurable timeout and error handling (`TamerTimeoutError`, `TamerUnsolvableError`, `TamerParseError`)

3. **Fast Downward Planner Module** — `mm_drrt/planner/pddl_planner.py` + `mm_drrt/planner/classical_pddl.py`
   - `PDDLPlanner`, the fallback if Tamer fails
   - `classical_pddl.py` generates MA-PDDL (`:multi-agent :concurrency-network`), compiles it to classical PDDL via `serialize_cn`, then invokes Fast Downward

4. **Problem Generator** — `mm_drrt/planner/pddl_problem_generator.py`
   - Converts MM-dRRT environment state to a UPF `Problem` instance (used by the Tamer path)
   - Bidirectional object mapping (env object id ↔ PDDL name)

5. **Plan Parser** — `mm_drrt/utils/pddl_parser.py`
   - Converts PDDL plans (UPF `TimeTriggeredPlan`/`SequentialPlan`, or raw Fast Downward plan text) to MM-dRRT format
   - Extracts: `plan`, `action_orders`, `obj_orders`, `init_order_constraints`

6. **Domain Builder** — `mm_drrt/planner/pddl_domain.py`
   - Programmatic domain creation using the UPF API, used by the Tamer path

## Installation

```bash
pip install -r requirements.txt
```

`requirements.txt` includes `unified_planning` and `up_tamer` (the Tamer planner backend). Fast Downward and the `universal-pddl-parser-multiagent` serializer are separate, not pip-installable — see the main `README.md`'s "PDDL Planner" section for build instructions.

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

2a. Tamer path (tried first)
    ↓
    generate_problem() → UPF Problem with boolean fluents
    ('obj-location', m, f) → set_initial_value(obj_location(m, f), True)
    ↓
    OneshotPlanner(name='tamer') → TimeTriggeredPlan

2b. Fast Downward path (fallback if Tamer fails)
    ↓
    generate_classical_problem_files() → MA-PDDL → serialize_cn → classical PDDL
    ↓
    run_fast_downward() → plan text

3. Plan Parsing
   ↓
   parse_pddl_plan()
   → Converts to MM-dRRT format:
     plan: {'a0': ('transit', robot, obj, None, table_0), ...}
     action_orders: {robot_0: ('a0', 'a1')}
     obj_orders: {obj_0: ['a1']}
     init_order_constraints: (...)

4. MM-dRRT Refinement (Unchanged)
   ↓
   Placement → Subgoal → Path computation → dRRT*
```

## Architecture

```
mm_drrt/
├── planner/
│   ├── tamer_pddl_planner.py     Tamer planner orchestrator (tried first)
│   ├── pddl_planner.py           Fast Downward planner orchestrator (fallback)
│   ├── classical_pddl.py         MA-PDDL generation + Fast Downward invocation
│   ├── pddl_domain.py            Programmatic PDDL 2.1 domain builder (for Tamer)
│   ├── pddl_problem_generator.py Environment → UPF Problem converter (for Tamer)
│   └── task_planner.py           MM-dRRT plan refinement (existing)
├── utils/
│   └── pddl_parser.py            PDDL plan → MM-dRRT converter
└── pddl/
    └── domains/
        └── mm_drrt_manipulation.pddl  PDDL 2.1 domain definition (reference copy)
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
        ('obj-location', self.m_objs[0], self.f_objs[0]),  # object m_objs[0] is on f_objs[0]
        ('obj-clear', self.m_objs[0]),
        ('surface-accessible', self.f_objs[0]),
        ('surface-accessible', self.f_objs[1])
    ]

    goal_state = [
        ('obj-location', self.m_objs[0], self.f_objs[1]),  # object m_objs[0] should end up on f_objs[1]
        ('robot-free', self.robots[0])
    ]

    return objects, init_state, goal_state
```

Every tuple is a boolean predicate — `(predicate_name, *args)` is set true in `init_state`, or required true in `goal_state`. `classical_pddl.py` additionally special-cases `obj-location` when compiling for Fast Downward (translating it to its own `at` predicate internally); no other predicate names are treated specially.

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
