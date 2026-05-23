(define (domain mm-drrt-manipulation)
  (:requirements :strips :typing :durative-actions :duration-inequalities :object-fluents)

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
  )

  (:functions
    (obj-location ?m - movable-obj) - fixed-obj
  )

  (:durative-action transit
    :parameters (?r - robot ?m - movable-obj ?from - fixed-obj)
    :duration (= ?duration 10)
    :condition (and
      (at start (robot-free ?r))
      (at start (= (obj-location ?m) ?from))
      (at start (obj-clear ?m))
      (over all (surface-accessible ?from))
      (over all (robot-can-reach ?r ?from))
    )
    :effect (and
      (at start (not (robot-free ?r)))
      (at start (not (obj-clear ?m)))
      (at end   (holding ?r ?m))
    )
  )

  (:durative-action transfer
    :parameters (?r - robot ?m - movable-obj ?to - fixed-obj)
    :duration (= ?duration 10)
    :condition (and
      (at start (holding ?r ?m))
      (over all (surface-accessible ?to))
      (over all (robot-can-reach ?r ?to))
    )
    :effect (and
      (at end (robot-free ?r))
      (at end (not (holding ?r ?m)))
      (at end (obj-clear ?m))
      (at end (assign (obj-location ?m) ?to))
    )
  )

)
