; Control for deadlock_demo_problem.pddl: same domain, but no circular wait -- robot0 starts
; holding table_A, table_B is left free rather than pre-claimed by robot1. robot0 can acquire
; table_B itself and finish; robot1 then acquires both once robot0 releases them. Confirms the
; UNSOLVABLE result in deadlock_demo_problem.pddl comes from the circular allocation, not from
; a bug in the domain's action definitions.
(define (problem deadlock-demo-control-problem)
  (:domain deadlock-demo)
  (:objects
    robot0 robot1 - robot
    table_A table_B - resource
  )
  (:init
    (holds robot0 table_A)
    (free table_B)
  )
  (:goal (and
    (done robot0)
    (done robot1)
  ))
)
