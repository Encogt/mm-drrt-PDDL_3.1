; Same circular-wait shape as deadlock_demo_problem.pddl, but using real object names from
; ExampleTwoRobotsRaiEnvironment (table_start, table_mid, robot0, robot1) so it passes
; pddl_file_planner.py's build_env_object_mapper() name-matching check and can be pointed at
; main_rai.py's --pddl_domain_file/--pddl_problem_file, not just the standalone solve_pddl.py
; sandbox. Domain logic is unaffected by naming, so Tamer's result is identical to
; deadlock_demo_problem.pddl: UNSOLVABLE.
;
; Run: python main_rai.py --use_gui --env_type exp_two_robots_rai --num_robots 2 --num_objs 1 \
;        --pddl_domain_file mm_drrt/pddl/domains/deadlock_demo_domain.pddl \
;        --pddl_problem_file mm_drrt/pddl/problems/deadlock_demo_env_problem.pddl
(define (problem deadlock-demo-env-problem)
  (:domain deadlock-demo)
  (:objects
    robot0 robot1 - robot
    table_start table_mid - resource
  )
  (:init
    (holds robot0 table_start)
    (holds robot1 table_mid)
  )
  (:goal (and
    (done robot0)
    (done robot1)
  ))
)
