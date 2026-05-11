(define (domain mm-drrt-manipulation)
  (:requirements :strips :typing :object-fluents)

  (:types
    robot
    movable-obj
    fixed-obj
  )

  (:predicates
    (robot-at-base ?r - robot)                     ; robot is at its base position
    (robot-free ?r - robot)                        ; robot is not holding anything
    (holding ?r - robot ?m - movable-obj)          ; robot holds object
    (obj-clear ?m - movable-obj)                   ; object can be grasped
    (surface-accessible ?f - fixed-obj)            ; surface is unblocked
    (robot-can-reach ?r - robot ?f - fixed-obj)    ; robot can physically reach surface
  )

  (:functions
    (obj-location ?m - movable-obj) - fixed-obj  ; surface the object rests on
  )

  (:action transit
    :parameters (?r - robot ?m - movable-obj ?from - fixed-obj)
    :precondition (and
      (robot-free ?r)
      (= (obj-location ?m) ?from)
      (obj-clear ?m)
      (surface-accessible ?from)
      (robot-can-reach ?r ?from)
    )
    :effect (and
      (holding ?r ?m)
      (not (robot-free ?r))
      (not (obj-clear ?m))
    )
  )

  (:action transfer
    :parameters (?r - robot ?m - movable-obj ?to - fixed-obj)
    :precondition (and
      (holding ?r ?m)
      (surface-accessible ?to)
      (robot-can-reach ?r ?to)
    )
    :effect (and
      (robot-free ?r)
      (not (holding ?r ?m))
      (obj-clear ?m)
      (assign (obj-location ?m) ?to)
    )
  )

)
