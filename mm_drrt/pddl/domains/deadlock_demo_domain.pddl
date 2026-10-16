; Minimal deadlock demo -- deliberately NOT durative-actions, to isolate the pure logical
; circular-wait question from the temporal-serialization question already answered for
; mm_drrt_manipulation.pddl (there, no action ever holds two `occupied` resources at once,
; so contention is always resolvable by picking *some* order -- see that domain's comments).
;
; Here, `finish` requires a robot to be holding BOTH resources simultaneously, and is the ONLY
; action that ever frees a resource. That creates a genuine two-resource circular wait if two
; robots start out holding one resource each and needing the other's: robot0 holds table_A and
; wants table_B, robot1 holds table_B and wants table_A, and nothing short of `finish` -- which
; neither can ever reach -- frees either one. This is the classical dining-philosophers /
; resource-allocation-graph deadlock, structurally analogous to what a bimanual/two-surface
; action would introduce into mm_drrt_manipulation.pddl if one were added.
(define (domain deadlock-demo)
  (:requirements :strips :typing)

  (:types
    robot
    resource
  )

  (:predicates
    (free ?x - resource)
    (holds ?r - robot ?x - resource)
    (done ?r - robot)
  )

  (:action acquire
    :parameters (?r - robot ?x - resource)
    :precondition (free ?x)
    :effect (and
      (not (free ?x))
      (holds ?r ?x)
    )
  )

  (:action finish
    :parameters (?r - robot ?x - resource ?y - resource)
    :precondition (and (holds ?r ?x) (holds ?r ?y))
    :effect (and
      (done ?r)
      (free ?x)
      (free ?y)
      (not (holds ?r ?x))
      (not (holds ?r ?y))
    )
  )
)
