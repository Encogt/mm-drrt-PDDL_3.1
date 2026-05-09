(define (domain mm-drrt-manipulation)
  (:requirements :strips :typing)

  (:types
    robot
    movable-obj
    fixed-obj
  )

  (:predicates
    (robot-at-base ?r - robot)            ; robot is at its base position
    (robot-free ?r - robot)               ; robot is not holding anything
    (holding ?r - robot ?m - movable-obj) ; robot holds object
    (obj-on ?m - movable-obj ?f - fixed-obj) ; object is on surface
    (obj-clear ?m - movable-obj)          ; object can be grasped
    (surface-accessible ?f - fixed-obj)   ; surface can be reached
  )

  (:action transit
    :parameters (?r - robot ?m - movable-obj ?from - fixed-obj)
    :precondition (and
      (robot-free ?r)
      (obj-on ?m ?from)
      (obj-clear ?m)
      (surface-accessible ?from)
    )
    :effect (and
      (holding ?r ?m)
      (not (robot-free ?r))
      (not (obj-on ?m ?from))
      (not (obj-clear ?m))
    )
  )

  (:action transfer
    :parameters (?r - robot ?m - movable-obj ?to - fixed-obj)
    :precondition (and
      (holding ?r ?m)
      (surface-accessible ?to)
    )
    :effect (and
      (robot-free ?r)
      (obj-on ?m ?to)
      (not (holding ?r ?m))
      (obj-clear ?m)
    )
  )

)

