; Minimal showcase for the minimal-temporal-constraint region encoding in
; mm_drrt_manipulation.pddl (transit-approach/-occupy/-depart, transfer-occupy/-depart): robot0
; and robot1 are each already holding a DIFFERENT object and both want to place it on the SAME
; shared surface, table_mid. No object links them (unlike two_robots_relay_problem.pddl's
; same-object handoff) -- the only thing that can force them apart is the region conflict on
; table_mid itself, so this isolates exactly the mechanism the new encoding changes.
;
; Compare against the OLD full-duration `occupied` mutex (git history, or copy this domain and
; revert transfer-occupy/-depart back into one plain `transfer` action) to see the effect:
;
;   OLD (full-duration mutex):
;     [ 0.00 -> 10.00]  transfer(robot0, box0, table_mid)
;     [10.01 -> 20.01]  transfer(robot1, box1, table_mid)      <- waits out robot0's WHOLE transfer
;
;   NEW (minimal region constraint, this domain):
;     [ 0.00 ->  7.00]  transfer-occupy(robot0, box0, table_mid)
;     [ 7.01 -> 10.01]  transfer-depart(robot0, box0, table_mid)
;     [ 7.01 -> 14.01]  transfer-occupy(robot1, box1, table_mid)   <- starts as soon as robot0's
;                                                                     OCCUPY phase ends, overlapping
;                                                                     robot0's depart phase
;     [14.02 -> 17.02]  transfer-depart(robot1, box1, table_mid)
;
; robot1's occupy phase only has to wait out robot0's occupy phase (si + beta_ik <= sj + alpha_jk),
; not robot0's full action -- a 3-unit makespan reduction (20.01 -> 17.02) from partial overlap
; that was structurally impossible under the old encoding.
;
; Run: python solve_pddl.py --problem mm_drrt/pddl/problems/region_overlap_demo_problem.pddl
(define (problem mm-drrt-region-overlap-demo)
  (:domain mm-drrt-manipulation)
  (:objects
    robot0 robot1 - robot
    box0 box1 - movable-obj
    table_mid - fixed-obj
  )
  (:init
    (holding robot0 box0)
    (holding robot1 box1)
    (obj-clear box0)
    (obj-clear box1)
    (surface-accessible table_mid)
    (robot-can-reach robot0 table_mid)
    (robot-can-reach robot1 table_mid)
  )
  (:goal (and
    (obj-location box0 table_mid)
    (obj-location box1 table_mid)
    (robot-free robot0)
    (robot-free robot1)
  ))
)
