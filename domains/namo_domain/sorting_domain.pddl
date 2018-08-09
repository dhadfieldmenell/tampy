(define (domain sorting_domain)
    (:predicates (CanAtTarget ?can - Can ?target - Target)
                 (CanInGripper ?can - Can)
    )

    (:types Can Target)

    (:action grasp
        :parameters (?can - Can)
        :precondition (forall (?c - Can) (not (CanInGripper ?c)))
        :effect (CanInGripper ?can)
    )

    (:action putdown
        :parameters (?can - Can ?target - Target)
        :precondition (CanInGripper ?can)
        :effect (and (not (CanInGripper ?can))
                     (CanAtTarget ?can ?target))
    )
)
