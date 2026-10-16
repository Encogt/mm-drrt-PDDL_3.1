; Circular-wait setup: robot0 already holds table_A and needs table_B to finish; robot1
; already holds table_B and needs table_A to finish. Neither `table_A` nor `table_B` is ever
; free again except via `finish`, which neither robot can reach without the other releasing
; first. No action ordering resolves this -- unlike mm_drrt_manipulation.pddl's shared-surface
; contention, which is always resolvable by *some* serial order.
;
; Run: python solve_pddl.py --domain mm_drrt/pddl/domains/deadlock_demo_domain.pddl \
;                            --problem mm_drrt/pddl/problems/deadlock_demo_problem.pddl
(define (problem deadlock-demo-problem)
  (:domain deadlock-demo)
  (:objects
    robot0 robot1 - robot
    table_A table_B - resource
  )
  (:init
    (holds robot0 table_A)
    (holds robot1 table_B)
  )
  (:goal (and
    (done robot0)
    (done robot1)
  ))
)
