; Real, hand-editable PDDL 2.1 problem for mm_drrt/pddl/domains/mm_drrt_manipulation.pddl --
; matches example_two_robots_rai_env.py's 1-object relay (l_panda: table_start -> table_mid,
; r_panda: table_mid -> table_end).
;
; Edit either this file or ../domains/mm_drrt_manipulation.pddl directly and re-run:
;   python solve_pddl.py
; to see how the solver reacts -- e.g.:
;   - change the domain's (= ?duration 10) to something else, or to a bound
;     (<= ?duration 5) / (>= ?duration 20) (the domain already declares
;     :duration-inequalities in its :requirements, so bounds are legal, not just fixed values)
;   - delete an (robot-can-reach ...) fact below to make the relay physically impossible
;   - add an unreachable (:goal ...) condition
;   - break the syntax entirely, to see the parser's error instead of a solver result
(define (problem mm-drrt-demo)
  (:domain mm-drrt-manipulation)
  (:objects
    robot0 robot1 - robot
    box0 - movable-obj
    table_start table_mid table_end - fixed-obj
  )
  (:init
    (robot-free robot0)
    (robot-free robot1)
    (obj-location box0 table_start)
    (obj-clear box0)
    (surface-accessible table_start)
    (surface-accessible table_mid)
    (surface-accessible table_end)
    (robot-can-reach robot0 table_start)
    (robot-can-reach robot0 table_mid)
    (robot-can-reach robot1 table_mid)
    (robot-can-reach robot1 table_end)
  )
  (:goal (and
    (obj-location box0 table_end)
    (robot-free robot0)
    (robot-free robot1)
    (robot-at-base robot0)
))
)
