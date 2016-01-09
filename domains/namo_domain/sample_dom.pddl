(define (domain robotics)
  (:requirements :strips :equality :typing)
  (:types Can Target Symbol Robot)

  (:predicates
   (At ?var0 - Can ?var1 - Target)
   (RobotAt ?var0 - Robot ?var1 - Symbol)
   (InGripper ?var0 - Can)
   (IsGP ?var0 - Symbol ?var1 - Can)
   (IsPDP ?var0 - Symbol ?var1 - Target)
   (Obstructs ?var0 - Robot ?var1 - Can)
   )

  (:action moveto
           :parameters (?robot - Robot ?start - Symbol ?end - Symbol)
           :precondition (and
                          (RobotAt ?robot ?start)
                          (forall (?obj - Can) (not (Obstructs ?robot ?obj)))
                          )
           :effect (and
                    (not (RobotAt ?robot ?start))
                    (RobotAt ?robot ?end)
                    )
           )

  (:action grasp
           :parameters (?robot - Robot ?can - Can ?target - Target ?gp - Symbol)
           :precondition (and
                          (At ?can ?target)
                          (RobotAt ?robot ?gp)
                          (IsGP ?gp ?can)
                          (forall (?obj - Can) (not (InGripper ?obj)))
                          (forall (?obj - Can) (not (Obstructs ?robot ?obj)))
                          )
           :effect (and
                    (not (At ?can ?target))
                    (InGripper ?can)
                    (not (Obstructs ?robot ?can))
                    )
           )

  (:action putdown
           :parameters (?robot - Robot ?can - Can ?target - Target ?pdp - Symbol)
           :precondition (and
                          (RobotAt ?robot ?pdp)
                          (IsPDP ?pdp ?target)
                          (InGripper ?can)
                          (forall (?obj - Can) (not (At ?obj ?target)))
                          (forall (?obj - Can) (not (Obstructs ?robot ?obj)))
                          )
           :effect (and
                    (At ?can ?target)
                    (not (InGripper ?can))
                    )
           )
  )
