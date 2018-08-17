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
        :precondition (and (CanInGripper ?can)
                           (forall (?c - Can) (not (CanAtTarget ?c ?target))))
        :effect (and (not (CanInGripper ?can))
                     (CanAtTarget ?can ?target))
    )
)
