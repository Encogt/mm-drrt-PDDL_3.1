# PDDL 3.1 Integration for MM-dRRT

This fork extends the original [MM-dRRT](https://github.com/syc7446/mm-drrt) with PDDL 3.1 planner integration for automatic task plan generation.

## Overview

The PDDL integration allows MM-dRRT to automatically generate task plan skeletons using standard PDDL planners, rather than requiring manual plan specification. The PDDL planner generates the high-level action sequence, and MM-dRRT refines it into executable motion plans.

## What's New

### Core Integration Components

1. **PDDL Domain Model** - `mm_drrt/pddl/domains/mm_drrt_manipulation.pddl`
   - STRIPS-based domain for robot manipulation
   - Actions: `transit` (pick), `transfer` (place)
   - Predicates: `robot-free`, `holding`, `obj-on`, `obj-clear`, `surface-accessible`

2. **PDDL Planner Module** - `mm_drrt/planner/pddl_planner.py`
   - Main orchestrator using Unified Planning Framework
   - Supports multiple planner backends (pyperplan, ENHSP, TAMER, etc.)
   - Automatic fallback to manual planning on failure
   - Configurable timeout and error handling

3. **Problem Generator** - `mm_drrt/planner/pddl_problem_generator.py`
   - Converts MM-dRRT environment state to PDDL problems
   - Bidirectional object mapping (PyBullet ↔ PDDL)
   - Automatic initial state and goal generation

4. **Plan Parser** - `mm_drrt/utils/pddl_parser.py`
   - Converts PDDL plans to MM-dRRT format
   - Extracts: `plan`, `action_orders`, `obj_orders`, `init_order_constraints`
   - Supports both sequential and temporal plans

5. **Domain Builder** - `mm_drrt/planner/pddl_domain.py`
   - Programmatic domain creation using UPF API
   - Easy to extend with new actions

### Modified Files

- **main.py** - Added `--use_pddl_planner` command-line flag
- **requirements.txt** - Added `unified-planning` dependency
- **examples/envs/example_single_robot_env.py** - Added `create_pddl_problem()` method

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/mm-drrt-pddl.git
cd mm-drrt-pddl
pip install -r requirements.txt
pip install up-pyperplan  # PDDL planner backend
```

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
┌─────────────────────────────────────────────────────────────┐
│                    PDDL Integration Pipeline                 │
└─────────────────────────────────────────────────────────────┘

1. Environment Specification
   ↓
   env.create_pddl_problem()
   → Returns: (objects, init_state, goal_state)

2. PDDL Problem Generation
   ↓
   PDDLProblemGenerator.generate_problem()
   → Creates UPF Problem instance
   → Maintains PyBullet ↔ PDDL object mapping

3. Planning
   ↓
   UPF Planner (pyperplan, ENHSP, etc.)
   → Finds satisficing or optimal plan

4. Plan Parsing
   ↓
   PDDLParser.parse_pddl_plan()
   → Converts to MM-dRRT format:
     - plan: {'a0': ('transit', robot, obj, None, table_0), ...}
     - action_orders: {robot_0: ('a0', 'a1')}
     - obj_orders: {obj_0: ['a1']}
     - init_order_constraints: ({'pre': 'a1', 'post': 'a9'}, ...)

5. MM-dRRT Refinement (Unchanged)
   ↓
   - Placement refinement
   - Subgoal refinement
   - Individual path computation
   - Composite path computation (dRRT*)
```

## Architecture

```
mm_drrt/
├── planner/
│   ├── pddl_planner.py          [NEW] Main PDDL planner orchestrator
│   ├── pddl_domain.py           [NEW] Programmatic domain builder
│   ├── pddl_problem_generator.py [NEW] Environment → PDDL converter
│   └── task_planner.py          [EXISTING] MM-dRRT plan refinement
├── utils/
│   └── pddl_parser.py           [NEW] PDDL plan → MM-dRRT converter
└── pddl/
    └── domains/
        └── mm_drrt_manipulation.pddl [NEW] PDDL domain definition
```

## Adding PDDL Support to Your Environment

To add PDDL planning support to a custom environment:

```python
def create_pddl_problem(self):
    """Define PDDL problem specification."""
    objects = {
        'robot': [self.robots[0], self.robots[1]],
        'movable-obj': [self.m_objs[0]],
        'fixed-obj': [self.f_objs[0], self.f_objs[1]]
    }

    init_state = [
        ('robot-free', self.robots[0]),
        ('robot-free', self.robots[1]),
        ('obj-on', self.m_objs[0], self.f_objs[0]),
        ('obj-clear', self.m_objs[0]),
        ('surface-accessible', self.f_objs[0]),
        ('surface-accessible', self.f_objs[1])
    ]

    goal_state = [
        ('obj-on', self.m_objs[0], self.f_objs[1])
    ]

    return objects, init_state, goal_state
```

## PDDL Planners

### Currently Supported

- **pyperplan** (default) - Lightweight Python STRIPS planner
  - Install: `pip install up-pyperplan`
  - Limitations: No temporal planning, no negative preconditions

### Future Support

For full PDDL 3.1 features (durative actions, temporal planning):
- **ENHSP** - Expressive Numeric Heuristic Search Planner
- **TAMER** - Temporal Action-based Modeler and Executor
- **POPF** - Partial Order Planning Forward-chainer

## Examples

### Single Robot Pick-and-Place

```bash
python main.py --use_pddl_planner \
               --env_type exp_single_robot \
               --num_robots 1 \
               --num_objs 1 \
               --use_gui
```

Expected output:
```
Using PDDL 3.1 planner to generate task plan...
✓ PDDL planner found satisficing solution
  Plan length: 2 actions
Successfully generated plan using PDDL planner
Step 1: placement refinement succeeded
Step 2: subgoal refinement succeeded
Step 3: individual path computation succeeded
Step 4: composite path computation succeeded
```

## Limitations

- Current domain uses STRIPS (not full PDDL 3.1 temporal planning)
- Pyperplan doesn't support negative preconditions
- For temporal/durative actions, switch to ENHSP or TAMER

## Future Enhancements

- [ ] Durative action support for true temporal planning
- [ ] Multi-robot PDDL problem generation
- [ ] Constraint-based temporal reasoning
- [ ] Integration with PDDLStream for combined task and motion planning
- [ ] Support for preferences and optimization metrics

## Citation

If you use this work, please cite both the original MM-dRRT paper and acknowledge this PDDL integration:

```bibtex
@inproceedings{sung2024mmdrrt,
  title={Asynchronous Task Plan Refinement for Multi-Robot Task and Motion Planning},
  author={Sung, Yoonchang and Shome, Rahul and Stone, Peter},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2024},
  organization={IEEE}
}
```

## License

This work extends the original MM-dRRT repository. Please refer to the original repository for licensing information.

## Acknowledgments

- Original MM-dRRT: [syc7446/mm-drrt](https://github.com/syc7446/mm-drrt)
- Unified Planning Framework: [aiplan4eu/unified-planning](https://github.com/aiplan4eu/unified-planning)
- Pyperplan: [aibasel/pyperplan](https://github.com/aibasel/pyperplan)
