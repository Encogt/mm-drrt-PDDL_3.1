(define (domain mm-drrt-manipulation)
  (:requirements :strips :typing :durative-actions :duration-inequalities)

  (:types
    robot
    movable-obj
    fixed-obj
  )

  (:predicates
    (robot-at-base ?r - robot)
    (robot-free ?r - robot)
    (holding ?r - robot ?m - movable-obj)
    (obj-clear ?m - movable-obj)
    (surface-accessible ?f - fixed-obj)
    (robot-can-reach ?r - robot ?f - fixed-obj)
    (obj-location ?m - movable-obj ?f - fixed-obj)
    ; True only while a robot's arm is ACTUALLY inside a surface's shared region right now --
    ; NOT held for an action's full duration any more (that was the old, coarser encoding: claimed
    ; "at start", released "at end", forcing any two actions on the same surface fully apart).
    ; transit only claims it 3 (of the total 10) time units into its motion -- the earlier
    ; "approach" phase below doesn't touch the surface's region yet -- and holds it through the
    ; natural end, since the robot is still departing WITH the object right up to :duration.
    ; transfer claims it from the very start (it arrives already holding the object) and releases
    ; it 7 time units in, once placement is done; the remaining "depart" phase has already left
    ; the region. This is the minimal temporal constraint: two robots conflicting in a shared
    ; region k only get forced apart across the sub-intervals [start_i + alpha_ik, start_i +
    ; beta_ik] that actually touch it (si + beta_ik <= sj + alpha_jk, or the reverse) -- not their
    ; whole high-level action -- so e.g. one robot's approach can still run in parallel with
    ; another robot's retreat from the same surface.
    (occupied ?f - fixed-obj)
    ; Chain markers linking a split action's two phases for one (robot, obj, surface) instance --
    ; see transit-approach/transit-occupy and transfer-occupy/transfer-depart below. Purely
    ; internal bookkeeping, never referenced outside this domain.
    (transit-entered ?r - robot ?m - movable-obj ?f - fixed-obj)
    (transfer-occupied-done ?r - robot ?m - movable-obj ?f - fixed-obj)
  )

  ; transit(r, m, from): pick object m from surface from. Split into an approach phase (outside
  ; the region, doesn't touch `occupied`) and an occupy phase (inside the region from then until
  ; the natural end). Together these are exactly one logical transit of :duration 3+7=10 --
  ; mm_drrt/utils/pddl_parser.py folds the two phases back into a single 'transit' entry before
  ; anything downstream (MM-dRRT's PlanSkeleton/dRRT*) ever sees them.
  (:durative-action transit-approach
    :parameters (?r - robot ?m - movable-obj ?from - fixed-obj)
    :duration (= ?duration 3)
    :condition (and
      (at start (robot-free ?r))
      (at start (obj-location ?m ?from))
      (at start (obj-clear ?m))
      (over all (surface-accessible ?from))
      (over all (robot-can-reach ?r ?from))
    )
    :effect (and
      (at start (not (robot-free ?r)))
      (at start (not (obj-clear ?m)))
      (at start (not (obj-location ?m ?from)))
      (at end   (transit-entered ?r ?m ?from))
    )
  )

  (:durative-action transit-occupy
    :parameters (?r - robot ?m - movable-obj ?from - fixed-obj)
    :duration (= ?duration 7)
    :condition (and
      (at start (transit-entered ?r ?m ?from))
      (over all (surface-accessible ?from))
      (over all (robot-can-reach ?r ?from))
      (at start (not (occupied ?from)))
    )
    :effect (and
      (at start (not (transit-entered ?r ?m ?from)))
      (at start (occupied ?from))
      (at end   (not (occupied ?from)))
      (at end   (holding ?r ?m))
    )
  )

  ; transfer(r, m, to): place object m on surface to. Split the OTHER way -- occupy starts
  ; immediately (the robot arrives already holding the object) and runs 7 of the total 10 time
  ; units, then a depart phase (outside the region) finishes the retreat over the remaining 3.
  (:durative-action transfer-occupy
    :parameters (?r - robot ?m - movable-obj ?to - fixed-obj)
    :duration (= ?duration 7)
    :condition (and
      (at start (holding ?r ?m))
      (over all (surface-accessible ?to))
      (over all (robot-can-reach ?r ?to))
      (at start (not (occupied ?to)))
      ; `holding` alone stays true for this whole transfer (occupy AND depart), so without this
      ; guard the search is free to fire a second, pointless transfer-occupy for the same (r, m,
      ; to) right after the first one ends but before its depart has consumed the marker below --
      ; confirmed via solve_pddl.py against the crossing-relay problem before this guard was added.
      (at start (not (transfer-occupied-done ?r ?m ?to)))
    )
    :effect (and
      (at start (occupied ?to))
      (at end   (not (occupied ?to)))
      (at end   (transfer-occupied-done ?r ?m ?to))
    )
  )

  (:durative-action transfer-depart
    :parameters (?r - robot ?m - movable-obj ?to - fixed-obj)
    :duration (= ?duration 3)
    :condition (and
      (at start (transfer-occupied-done ?r ?m ?to))
      (over all (surface-accessible ?to))
      (over all (robot-can-reach ?r ?to))
    )
    :effect (and
      (at start (not (transfer-occupied-done ?r ?m ?to)))
      (at end   (robot-free ?r))
      (at end   (not (holding ?r ?m)))
      (at end   (obj-clear ?m))
      (at end   (obj-location ?m ?to))
    )
  )

)
