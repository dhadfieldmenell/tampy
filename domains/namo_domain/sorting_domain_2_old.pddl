(define (domain sorting_domain)
    (:requirements :equality)
    (:predicates (CanAtTarget ?can - Can ?target - Target)
                 (CanInGripper ?can - Can)
                 (CanObstructs ?can1 - Can ?can2 - Can) ; can1 obstructs the path to can2
                 (CanObstructsTarget ?can - Can ?target - Target)
                 (CanInReach ?can - Can)
                 (NearCan ?can - Can)
                 (WaitingOnCan ?waiter - Can ?obstr - Can) ; waiter cannot move until obstr has been moved
                 (WaitingOnTarget ?waiter - Can ?obstr - Target)
    )

    (:types Can Target)

    (:action move_can
        :parameters (?can - Can ?target - Target)
        :precondition (and (forall (?c - Can) (not (CanObstructs ?c ?can)))
                           (forall (?c - Can) (not (CanObstructsTarget ?c ?target)))
                           (forall (?c - Can) (not (CanAtTarget ?c ?target)))
                           (forall (?c - Can) (not (WaitingOnCan ?can ?c)))
                           (forall (?t - Target) (not (WaitingOnTarget ?can ?t))))
        :effect (and (CanAtTarget ?can ?target)
                     (forall (?c - Can) (when (CanObstructs ?can ?c) (WaitingOnCan ?can ?c))) ; when can was obstructing c, cannot move can again until c moves
                     (forall (?t - Target) (when (CanObstructsTarget ?can ?t) (WaitingOnTarget ?can ?t)))
                     (forall (?c - Can) (not (CanObstructs ?can ?c)))
                     (forall (?c - Can) (not (WaitingOnCan ?c ?can)))
                     (forall (?c - Can) (not (WaitingOnTarget ?c ?target)))
                     (forall (?t - Target) (when (not (= ?t ?target)) (not (CanAtTarget ?can ?t))))
                     (forall (?t - Target) (when (not (= ?t ?target)) (not (CanObstructsTarget ?can ?t)))))
    )

)
