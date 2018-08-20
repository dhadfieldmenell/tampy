(define (domain sorting_domain)
    (:predicates (CanAtTarget ?can - Can ?target - Target)
                 (CanInGripper ?can - Can)
                 (CanObstructs ?can1 - Can ?can2 - Can)
    )

    (:types Can Target)

    (:action grasp
        :parameters (?can - Can)
        :precondition (and (forall (?c - Can) (not (CanInGripper ?c)))
                           (forall (?c - Can) (not (CanObstructs ?c ?can))))
        :effect (CanInGripper ?can)
    )

    (:action putdown
        :parameters (?can - Can ?target - Target)
        :precondition (and (CanInGripper ?can)
                           (forall (?c - Can) (not (CanAtTarget ?c ?target))))
        :effect (and (not (CanInGripper ?can))
                     (CanAtTarget ?can ?target)
                     (forall (?c - Can) (not (CanObstructs ?can ?c))))
    )
)
