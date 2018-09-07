(define (domain sorting_domain)
    (:requirements :equality)
    (:predicates (ClothAtTarget ?cloth - Cloth ?target - Target)
                 (ClothInGripperLeft ?cloth - Cloth)
                 (ClothInGripperRight ?cloth - Cloth)
                 (ClothInReach ?cloth - Cloth)
                 (TargetIsOnLeft ?target - Target)
                 (TargetIsOnRight ?target - Target)
                 (ClothIsOnLeft ?cloth - Cloth)
                 (ClothIsOnRight ?cloth - Cloth)
                 (WasherOpen ?washer - Washer)
                 (WasherClosed ?washer - Washer)
                 (ClothInWasher ?cloth - Cloth ?washer - Washer)
    )

    (:types Cloth Target Washer)

    (:action grasp_left
        :parameters (?cloth - Cloth)
        :precondition (and (forall (?w - Washer) (not (ClothInWasher ?w)))
                           (forall (?c - Cloth) (not (ClothInGripperLeft ?c))))
        :effect (ClothInGripperLeft ?cloth)
    )

    (:action putdown_left
        :parameters (?cloth - Cloth ?target - Target)
        :precondition (and (ClothInGripperLeft ?cloth)
                           (forall (?c - Cloth) (not (ClothAtTarget ?c ?target))))
        :effect (and (not (ClothInGripperLeft ?cloth))
                     (ClothAtTarget ?cloth ?target)
                     (forall (?t - Target) (when (not (= ?t ?target)) (not (ClothAtTarget ?cloth ?t)))))
    )

    (:action open_washer
        :parameters (?washer - Washer)
        :precondition (and (forall (?c - Cloth) (not (ClothInGripperLeft ?c)))
                           (not (WasherOpen ?washer)))
        :effect (WasherOpen ?washer)
    )

    (:action close_washer
        :parameters (?washer - Washer)
        :precondition (and (forall (?c - Cloth) (not (ClothInGripperLeft ?c)))
                           (not (WasherClosed ?washer)))
        :effect (WasherClosed ?washer)
    )

    (:action put_into_washer
        :parameters (?cloth - Cloth ?washer - Washer)
        :precondition (and (ClothInGripperLeft ?cloth)
                           (WasherOpen ?washer))
        :effect (and (ClothInWasher ?cloth ?washer)
                     (forall (?t - Target) (not (ClothAtTarget ?t))))
    )

    (:action take_out_of_washer
        :parameters (?cloth - Cloth ?washer - Washer)
        :precondition (and (ClothInWasher ?cloth ?washer)
                           (WasherOpen ?washer)
                           (forall (?c - Cloth) (not (ClothInGripperLeft ?c)))
        :effect (and (not (ClothInWasher ?cloth ?washer))
                     (ClothInGripperLeft ?cloth))
    )
)
