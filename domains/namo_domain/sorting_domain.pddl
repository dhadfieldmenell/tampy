(define (domain sorting_domain)
    (:predicates (CanAtTarget ?can ?target)
                 (CanInGripper ?can)
    )

    (:types Can Target)

    (:action grasp
        :paramters (?can - Can)
        :pecondition ((forall ?c - Can) (not (CanInGripper ?c)))
        :effect (CanInGripper ?can)
    )

    (:action putdown
        :parameters (?can - Can ?target - Target)
        :precondition (CanInGripper ?can)
        :effect (and (not (CanInGripper ?can))
                     (CanAtTarget ?can ?target))
    )
)
