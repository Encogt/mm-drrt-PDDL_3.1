# MM-dRRT

This repository is designed for MM-dRRT, a multi-robot task and motion planning algorithm implemented in the PyBullet simulator. Please refer to the following paper for more details, and consider citing this work if you find this repository useful.
```
Yoonchang Sung, Rahul Shome, and Peter Stone. "Asynchronous Task Plan Refinement for Multi-Robot Task and Motion Planning." In 2024 IEEE International Conference on Robotics and Automation (ICRA), IEEE, 2024.
```


## Installation
```
git clone --recursive https://github.com/syc7446/mm-drrt.git
cd mm-drrt/
pip install -r requirements.txt
python -m main
```

### PDDL Planner (optional)

To use `--use_pddl_planner`, two additional tools must be built from source.

**1. Fast Downward**
```
git clone https://github.com/aibasel/downward.git ~/fast-downward
cd ~/fast-downward
python3 build.py
```

**2. universal-pddl-parser-multiagent** (serializes MA-PDDL to classical PDDL)
```
git clone --recursive https://github.com/aig-upf/universal-pddl-parser-multiagent.git ~/universal-pddl-parser-multiagent
cd ~/universal-pddl-parser-multiagent/universal-pddl-parser
scons
cd ..
scons examples/serialize_cn
```

If `scons` is not installed: `pip install scons`

The code finds both tools automatically if they are at the default paths above. To use custom locations, set these environment variables:
```
export FAST_DOWNWARD_CMD=/path/to/fast-downward.py
export UPDDL_SERIALIZER_CMD=/path/to/serialize.bin
```

## Inputs and Parameters
### Inputs
All the relevant files are included in the `/examples/envs/` folder. Find the `create_plan_order_constraints` function in the environment class, where inputs for a problem instance can be specified.
- **plan**: All the abstract actions are specified as follows. {action name (e.g. 'a0'): (robot ID, movable object ID, fixed object ID (move from), fixed object ID (move to))}
- **action_orders**: The sequence of abstract actions for each robot is specified as follows. {robot ID: (sequence of action names (e.g. 'a0', 'a1', 'a4', 'a6'))}
- **obj_orders**: The sequence of abstract actions for each movable object is specified as follows. {movable object ID: [sequence of action names (e.g. 'a1', 'a5')]}
- **init_order_constraints**: All the temporal constraints among different robots' abstract actions are specified. For example, Robot 1 must pick up Movable object 2 before Robot 2 picks up Movable object 3 from the same Fixed object, such as a table. {'pre': action name, 'post': action name}

### Parameters
- **num_robots**: The total number of robots used.
- **num_objs**: The total number of movable objects.
- **num_placement_samples**: The total number of samples used for the Place action.
- **num_base_samples**: The total number of samples used for solving a base motion planning problem. For example, if PRM is used, this parameter specifies the total number of samples used to construct the roadmap.
- **num_arm_samples**: This parameter is equivalent to **num_base_samples** but for arm motions.
- **env_type**: This parameter determines which example to run.
- **use_gui**: This parameter determines whether to use a GUI for visualization.

## Examples
We provide two example codes to facilitate understanding of how MM-dRRT works. The first example involves a single robot picking up an object from one table and placing it on another. The second example involves two robots moving two objects located on two tables.

- Single robot, single object: Set both `num_robots` and `num_objs` to 1. Set `env_type` to `exp_single_robot`. 
- Two robots, two objects: Set both `num_robots` and `num_objs` to 2. Set `env_type` to `exp_two_robots`. 

In both examples, focus on the `_create_problem` function in `example_single_robot_env` and `example_two_robots_env` files located in the `/examples/envs/` folder to understand how the environments are specified.