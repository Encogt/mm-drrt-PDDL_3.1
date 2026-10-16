; 2-object crossing relay -- matches example_two_robots_rai_env.py's num_objs=2 scenario:
; box0 goes table_start -> table_mid -> table_end (robot0 hands off to robot1), box1 goes
; table_end -> table_mid -> table_start (robot1 hands off to robot0).
;
; Unlike two_robots_relay_problem.pddl's single-object relay, box0's chain and box1's chain
; share NO object -- so nothing causally forces robot0's transfer(box0 -> table_mid) and
; robot1's transfer(box1 -> table_mid) apart. Before the domain's `occupied` mutex, Tamer was
; free to schedule those fully in parallel; now it must serialize every use of table_mid, and
; exactly how far apart those windows land is a direct function of the domain's :duration
; values -- run this through solve_pddl.py after changing transit/transfer :duration in
; mm_drrt_manipulation.pddl and watch the timeline shift.
(define (problem mm-drrt-crossing-relay)
  (:domain mm-drrt-manipulation)
  (:objects
    robot0 robot1 - robot
    box0 box1 - movable-obj
    table_start table_mid table_end - fixed-obj
  )
  (:init
    (robot-free robot0)
    (robot-free robot1)
    (obj-location box0 table_start)
    (obj-location box1 table_end)
    (obj-clear box0)
    (obj-clear box1)
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
    (obj-location box1 table_start)
    (robot-free robot0)
    (robot-free robot1)
  ))
)
