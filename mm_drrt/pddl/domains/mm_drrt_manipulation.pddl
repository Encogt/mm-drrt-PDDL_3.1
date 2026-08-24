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
    ; True while some robot is actively transiting-from or transferring-to ?f -- i.e. its arm is
    ; physically occupying that surface's workspace right now. Held true for the action's FULL
    ; duration (start to end, via the "at start"/"at end" effects below), not just an instant --
    ; so a robot may not even START using a surface until the previous user's ENTIRE declared
    ; action duration has elapsed. Guarded as an "at start" precondition, not "over all": an
    ; "over all (not (occupied ?f))" condition would be checked against the action's OWN "at
    ; start (occupied ?f)" effect and make every action unsolvable against itself (confirmed --
    ; that was tried first and made the domain permanently UNSOLVABLE_INCOMPLETELY for even the
    ; trivial 1-object relay). "at start" only checks the state just BEFORE this action's own
    ; start effects apply, so it correctly blocks a DIFFERENT action from starting while this one
    ; still holds the mutex, without conflicting with its own claim.
    ;
    ; This is what makes :duration matter: two robots sharing one surface (e.g. table_mid in a
    ; 2-object crossing relay, where no single object's own transit/transfer chain links them)
    ; must now be serialized by the solver, and how far apart their time windows land is a direct
    ; function of :duration -- previously nothing in this domain depended on the specific
    ; duration VALUE at all, only on causal (holding/obj-location) ordering.
    (occupied ?f - fixed-obj)
  )

  (:durative-action transit
    :parameters (?r - robot ?m - movable-obj ?from - fixed-obj)
    :duration (= ?duration 10)
    :condition (and
      (at start (robot-free ?r))
      (at start (obj-location ?m ?from))
      (at start (obj-clear ?m))
      (over all (surface-accessible ?from))
      (over all (robot-can-reach ?r ?from))
      (at start (not (occupied ?from)))
    )
    :effect (and
      (at start (not (robot-free ?r)))
      (at start (not (obj-clear ?m)))
      (at start (not (obj-location ?m ?from)))
      (at start (occupied ?from))
      (at end   (not (occupied ?from)))
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
      (at start (not (occupied ?to)))
    )
    :effect (and
      (at start (occupied ?to))
      (at end (not (occupied ?to)))
      (at end (robot-free ?r))
      (at end (not (holding ?r ?m)))
      (at end (obj-clear ?m))
      (at end (obj-location ?m ?to))
    )
  )

)
