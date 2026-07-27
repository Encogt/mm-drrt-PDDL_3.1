import numpy as np
import math
import time
import copy

from mm_drrt.utils.motion_planner_utils import OptimalNode, get_angle, \
    is_duplicate, get_sub_nodes, get_parent_node_index, get_local_paths, is_local_collision, \
    get_sub_q_near, connect_to_target, get_collision_free_paths_to_target, get_sub_q_list, \
    get_min_heuristic_vertex, get_random_neighbor_vertex, get_neighbor_vertices, compute_local_dist, \
    get_q_best, get_node_from_tree, rewire, get_heuristic_val, is_goal_in_tree, get_subprob, \
    get_substarts_subgoals, update_subprob_id, is_q_new_in_subgoal, get_goals, get_sub_samples, get_subattachments,\
    apply_order_constraints, is_violate_order_constraints, debug_2d_sub_sampling

from mm_drrt.utils.rai_motion_planner_utils import get_inter_robots_collision_fn


def _advance_past_trivial(sub_q, sub_goals, subprob_id, goals, joint_dim, roadmaps):
    """update_subprob_id, but also auto-skips any newly-entered placeholder roadmap segment whose
    initial_conf == final_conf (get_trivial_roadmap() in rai_motion_planner_utils.py -- stands in
    for a fixed-arm robot's non-existent base phase, so every 3-roadmap-per-action group the
    multi-robot indexing assumes still has 3 entries). Those roadmaps have exactly one vertex and
    no edges, so the normal search loop can never find a *different* neighbor to explore there
    (get_random_neighbor_vertex always returns the same isolated vertex) and would spin forever.
    sub_q_last stays valid as "current position" across a whole chain of these, since
    get_trivial_roadmap() always uses the SAME conf (a robot's carry_conf) for both its
    initial_conf and final_conf, matching the neighboring real segments' boundary value."""
    sub_q_last, sub_goals = update_subprob_id(sub_q, sub_goals, subprob_id, goals, joint_dim, roadmaps)
    advanced = True
    while advanced:
        advanced = False
        for r in range(len(roadmaps)):
            if subprob_id[r] >= len(roadmaps[r]):
                continue
            rm = roadmaps[r][subprob_id[r]]
            if (rm.initial_conf == rm.final_conf and sub_q_last[r] == sub_goals[r]
                    and len(roadmaps[r]) - 1 > subprob_id[r]):
                subprob_id[r] += 1
                sub_goals[r] = get_sub_q_list(get_substarts_subgoals(goals, subprob_id, joint_dim), len(roadmaps))[r]
                advanced = True
    return sub_q_last, sub_goals

MAX_DISTANCE = 0.0  # unused broad-phase margin in the RAI collision backend; kept for signature compat

class dRRTStar:

    def __init__(self, robots, joints, roadmaps, order_constraints, num_robots=1, heuristic_vals=[], radius=None,
                 attachments=[], max_distance=MAX_DISTANCE, use_aabb=False, cache=True):
        self.robots = robots
        self.joints = joints
        self.joint_dim = len(self.joints[0])
        self.roadmaps = roadmaps
        self.order_constraints = order_constraints
        self.num_robots = num_robots
        self.heuristic_vals = heuristic_vals
        self.radius = radius

        self.starts = []
        self.goals = []
        self.subprob_id = []
        for r in range(self.num_robots):
            starts = ()
            goals = ()
            for roadmap in self.roadmaps[r]:
                starts += roadmap.initial_conf
                goals += roadmap.final_conf
            self.starts.append(starts)
            self.goals.append(goals)
            # Skip past any LEADING trivial (start==goal) placeholder roadmap segments -- e.g.
            # get_trivial_roadmap()'s stand-in for a fixed-arm robot's non-existent base phase --
            # independently per robot, before the search even starts. This has to happen here
            # rather than only via _advance_past_trivial() during grow(): if EVERY robot's very
            # first segment happens to be trivial (true whenever every robot's first action is a
            # fresh pick, as in a relay), no robot can ever produce a q_new different from
            # sub_q_near (get_random_neighbor_vertex always returns the same isolated vertex for
            # all of them simultaneously), so the "prevent same" search loop can never break to
            # begin with -- there is no order-constraint concern in skipping this, since nothing
            # physical happens for a placeholder segment.
            start_id = 0
            while (start_id < len(self.roadmaps[r]) - 1
                  and self.roadmaps[r][start_id].initial_conf == self.roadmaps[r][start_id].final_conf):
                start_id += 1
            self.subprob_id.append(start_id)

        self.inter_robots_collision_fn = get_inter_robots_collision_fn(self.robots, self.joints, num_robots=num_robots)
        self.nodes = [OptimalNode(get_substarts_subgoals(self.starts, self.subprob_id, self.joint_dim),
                                  num_robots=self.num_robots, subprob_id=self.subprob_id, path=[],
                                  attachments=[None for _ in range(self.num_robots)])]

    def grow(self, num_iters=10, start_time=0., time_limit=math.inf, use_debug_plot=False,
             use_debug_verbal=False, debug_robot_id=0):
        use_debug_verbal = False
        best_paths = []
        best_path_cost = float('inf')
        sub_q_last = get_sub_q_list(get_substarts_subgoals(self.starts, self.subprob_id, self.joint_dim), self.num_robots)
        sub_goals = get_sub_q_list(get_substarts_subgoals(self.goals, self.subprob_id, self.joint_dim), self.num_robots)
        is_found_path = False
        loop_count = 0

        while time.time() < start_time + time_limit:
            start = time.time()
            loop_count += 1
            if loop_count == 1 or loop_count % 25 == 0:
                elapsed = time.time() - start_time
                print(f"Step 4: dRRT* search loop {loop_count}, elapsed {elapsed:.2f}s, nodes={len(self.nodes)}, subprob={self.subprob_id}")
            if is_found_path: break
            if use_debug_verbal: print('New loop starts. Time taken so far: %.2fs' % (time.time() - start_time))
            for i in range(num_iters):
                if is_found_path: break
                if self.subprob_id == [6, 6]:
                    a = 1
                if sub_q_last is None:
                    sub_samples = []
                    for r in range(self.num_robots):
                        # sub_samples.append(get_subprob(self.roadmaps, self.subprob_id)[r].sub_sample_fn())
                        sub_samples = get_sub_samples(get_subprob(self.roadmaps, self.subprob_id)[r], sub_samples)
                    sub_q_near = get_sub_q_near(self.nodes, get_subprob(self.roadmaps, self.subprob_id), sub_samples, self.subprob_id)
                else:
                    sub_samples = sub_goals
                    sub_q_near = sub_q_last

                # get_random_neighbor_vertex(roadmap, q_near) always includes q_near itself as a
                # candidate, so if q_near's roadmap vertex happens to have zero edges (possible
                # with a sparse/small roadmap -- more likely with more robots, since every robot's
                # vertex must independently be non-isolated for this loop to ever break), this spins
                # forever. Cap it: if no robot's contribution can move off sub_q_near after enough
                # tries, give up on this sample and let the outer loop draw a fresh one next time.
                spin_cap_hit = False
                for _spin in range(2000):
                    q_new = ()
                    for r in range(self.num_robots):
                        if sub_samples[r] == sub_goals[r]:
                            temp_q_new = get_min_heuristic_vertex(get_subprob(self.roadmaps, self.subprob_id)[r],
                                                              sub_q_near[r], get_subprob(self.heuristic_vals, self.subprob_id)[r])
                        else:
                            temp_q_new = get_random_neighbor_vertex(get_subprob(self.roadmaps, self.subprob_id)[r], sub_q_near[r])
                        q_new += apply_order_constraints(temp_q_new, sub_q_near[r], r, self.order_constraints[r],
                                                         get_sub_q_list(get_substarts_subgoals(self.goals, self.subprob_id, self.joint_dim), self.num_robots),
                                                         self.subprob_id)
                    if get_sub_q_list(q_new, self.num_robots) == sub_q_near:
                        if use_debug_verbal: print('q_new is the same as sub_q_near')
                    else:
                        break
                else:
                    spin_cap_hit = True
                if spin_cap_hit:
                    sub_q_last = None
                    continue

                neighbor_nodes = get_neighbor_vertices(self.nodes, get_subprob(self.roadmaps, self.subprob_id), q_new, self.subprob_id)
                q_best, local_dist, local_paths = get_q_best(self.nodes, get_subprob(self.roadmaps, self.subprob_id), self.num_robots,
                                                             self.inter_robots_collision_fn, neighbor_nodes, q_new, self.subprob_id)
                if not q_best:
                    if use_debug_verbal: print('q_best is empty.')
                    sub_q_last = None
                    continue

                if best_paths:
                    if get_node_from_tree(self.nodes, q_best, self.subprob_id).cost + local_dist > best_path_cost:
                        if use_debug_verbal: print('Cost of q_new is larger than that of the best path.')
                        sub_q_last = None
                        continue

                is_update_subprob_id = False
                if not is_duplicate(self.nodes, q_new, self.subprob_id):
                    subprob_id = copy.deepcopy(self.subprob_id)
                    if is_q_new_in_subgoal(get_subprob(self.roadmaps, self.subprob_id), q_new,
                                           get_substarts_subgoals(self.goals, self.subprob_id, self.joint_dim)):
                        if use_debug_verbal: print('Some of robots reached their subgoals.')
                        sub_q_last, sub_goals = _advance_past_trivial(q_new, sub_goals, self.subprob_id,
                                                                  self.goals, self.joint_dim, self.roadmaps)
                        start = time.time()
                        is_update_subprob_id = True

                    if use_debug_verbal: print('New node is added to the tree.')
                    # remember that the local path is the path reaching to the current node
                    self.nodes.append(OptimalNode(q_new, num_robots=self.num_robots, d=local_dist,
                                                  parent=get_node_from_tree(self.nodes, q_best, subprob_id),
                                                  subprob_id=self.subprob_id, path=local_paths,
                                                  attachments=get_subattachments(self.roadmaps, self.subprob_id, self.nodes)))

                    # TODO: currently best path can be computed only when the final goals are reached. we do not consider optimality yet.
                    if is_goal_in_tree(self.nodes, get_goals(self.goals, self.joint_dim), self.roadmaps, self.subprob_id):
                        goal_node = get_node_from_tree(self.nodes, get_substarts_subgoals(self.goals, self.subprob_id,
                                                                                          self.joint_dim), self.subprob_id)
                        best_paths = goal_node.retrace()
                        best_path_cost = goal_node.cost
                        is_found_path = True
                else:
                    if use_debug_verbal: print('Rewiring: q_new is already in the tree.')
                    if q_best == q_new:
                        if use_debug_verbal: print('q_new == q_best so skip rewiring.')
                    # else:
                    #     rewire(self.nodes, get_subprob(self.roadmaps, self.subprob_id), self.num_robots, q_best, q_new)

                # for n in neighbor_nodes:
                #     local_paths = get_local_paths(roadmap=get_subprob(self.roadmaps, self.subprob_id), num_robots=self.num_robots,
                #                                   start_configs=get_sub_q_list(q_new, self.num_robots),
                #                                   target_configs=get_sub_q_list(n, self.num_robots))
                #     if get_node_from_tree(self.nodes, q_new).cost + compute_local_dist(get_subprob(self.roadmaps, self.subprob_id), q_new, n) < \
                #         get_node_from_tree(self.nodes, n).cost and \
                #         not is_local_collision(local_paths, self.inter_robots_collision_fn):
                #         if use_debug_verbal: print('Rewiring: neighbor nodes.')
                #         if n == q_new:
                #             if use_debug_verbal: print('q_new == q_neighbor so skip rewiring.')
                #         else:
                #             rewire(self.nodes, get_subprob(self.roadmaps, self.subprob_id), self.num_robots, q_new, n)

                if not is_update_subprob_id:
                    if get_heuristic_val(get_subprob(self.roadmaps, self.subprob_id), q_new,
                                         get_subprob(self.heuristic_vals, self.subprob_id)) < \
                            get_heuristic_val(get_subprob(self.roadmaps, self.subprob_id), q_best,
                                              get_subprob(self.heuristic_vals, self.subprob_id)):
                        sub_q_last = get_sub_q_list(q_new, self.num_robots)
                    else:
                        sub_q_last = None
                if time.time() - start_time > time_limit: break

            # Connect to target
            if use_debug_verbal: print('Trying to connect to the target goal. ')
            sub_goals = get_sub_q_list(get_substarts_subgoals(self.goals, self.subprob_id, self.joint_dim), self.num_robots)
            sub_q_near = get_sub_q_near(self.nodes, get_subprob(self.roadmaps, self.subprob_id), sub_goals, self.subprob_id)
            if sub_q_near == sub_goals:
                # Already at this subprob's goal without ever having to search for it -- e.g. a
                # trivial start==goal placeholder roadmap (get_trivial_roadmap in
                # rai_motion_planner_utils.py, used for a fixed-arm robot's non-existent "base"
                # phase). The normal advance-subprob_id path below only fires after new search
                # progress; replicate it here (with a zero-length "already there" local path) so
                # subprob_id still advances instead of stalling here forever.
                if use_debug_verbal: print('Goal has already reached.')
                if not is_violate_order_constraints(self.order_constraints, self.subprob_id):
                    parent_node_index = get_parent_node_index(self.nodes, sub_q_near)
                    local_paths = [[c] for c in sub_q_near]
                    node_pose = get_substarts_subgoals(self.goals, self.subprob_id, self.joint_dim)
                    sub_q_last, sub_goals = _advance_past_trivial(node_pose, sub_goals, self.subprob_id,
                                                              self.goals, self.joint_dim, self.roadmaps)
                    self.nodes.append(OptimalNode(node_pose, num_robots=self.num_robots, d=0,
                                                  parent=self.nodes[parent_node_index],
                                                  subprob_id=self.subprob_id, path=local_paths,
                                                  attachments=get_subattachments(self.roadmaps, self.subprob_id, self.nodes)))
                    if is_goal_in_tree(self.nodes, get_goals(self.goals, self.joint_dim), self.roadmaps, self.subprob_id):
                        goal_node = get_node_from_tree(self.nodes, get_substarts_subgoals(self.goals, self.subprob_id,
                                                                                          self.joint_dim), self.subprob_id)
                        best_paths = goal_node.retrace()
                        best_path_cost = goal_node.cost
                        is_found_path = True
            else:
                if not is_violate_order_constraints(self.order_constraints, self.subprob_id):
                    parent_node_index = get_parent_node_index(self.nodes, sub_q_near)
                    # in connect_to_target, collisions with obstacles are only checked. inter-robot collisions are checked only for starts and goals
                    local_paths = connect_to_target(roadmap=get_subprob(self.roadmaps, self.subprob_id),
                                                    num_robots=self.num_robots,
                                                    start_configs=self.nodes[parent_node_index].sub_config,
                                                    target_configs=sub_goals,
                                                    drrt_collision_fn=self.inter_robots_collision_fn)
                    if not local_paths:
                        if use_debug_verbal: print('Connecting to target is in collision with obstacles.')
                    else:
                        # in get_collision_free_paths_to_target, collisions with other robots are checked
                        local_paths = get_collision_free_paths_to_target(local_paths, self.inter_robots_collision_fn)
                        if use_debug_verbal: print('New node is added to the tree.')
                        dist_to_goal = compute_local_dist(get_subprob(self.roadmaps, self.subprob_id),
                                                          self.nodes[parent_node_index].config,
                                                          get_substarts_subgoals(self.goals, self.subprob_id, self.joint_dim))
                        node_pose = get_substarts_subgoals(self.goals, self.subprob_id, self.joint_dim)
                        sub_q_last, sub_goals = _advance_past_trivial(get_substarts_subgoals(self.goals, self.subprob_id, self.joint_dim),
                                                                  sub_goals, self.subprob_id, self.goals, self.joint_dim, self.roadmaps)
                        # print('subprob_id', self.subprob_id)
                        self.nodes.append(OptimalNode(node_pose, num_robots=self.num_robots, d=dist_to_goal,
                                                      parent=self.nodes[parent_node_index],
                                                      subprob_id=self.subprob_id, path=local_paths,
                                                      attachments=get_subattachments(self.roadmaps, self.subprob_id, self.nodes)))

                        # TODO: currently best path can be computed only when the final goals are reached. we do not consider optimality yet.
                        if is_goal_in_tree(self.nodes, get_goals(self.goals, self.joint_dim), self.roadmaps, self.subprob_id):
                            goal_node = get_node_from_tree(self.nodes, get_substarts_subgoals(self.goals, self.subprob_id,
                                                                                              self.joint_dim), self.subprob_id)
                            best_paths = goal_node.retrace()
                            best_path_cost = goal_node.cost
                            is_found_path = True
                        # best_paths, best_path_cost = self.update_best_path(best_paths, best_path_cost)
                else:
                    # if the algorithm gets stuck with this message, that highly implies that the roadmap size is insufficient, so feasible solution may not exist at all
                    if use_debug_verbal: print('order constraints are violated so skip connect_to_target')
            if time.time() - start_time > time_limit:
                if use_debug_verbal: print('Time out.')
                break

        if best_paths: print('dRRT* is solved successfully.')
        else:
            raise SystemExit('TIMEOUT: dRRT* is NOT solved.')
        return get_node_from_tree(self.nodes, get_substarts_subgoals(self.goals, self.subprob_id, self.joint_dim), self.subprob_id).retrace()

    def update_best_path(self, best_paths, best_path_cost):
        if is_goal_in_tree(self.nodes, get_goals(self.goals, self.joint_dim), self.roadmaps, self.subprob_id): # TODO: currently best path can be computed only when the final goals are reached.
            goal_node = get_node_from_tree(self.nodes, get_substarts_subgoals(self.goals, self.subprob_id, self.joint_dim), self.subprob_id)
            if goal_node.cost < best_path_cost:
                best_paths = goal_node.retrace()
                best_path_cost = goal_node.cost
        return best_paths, best_path_cost
