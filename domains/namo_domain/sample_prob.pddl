(define (problem sample_prob) (:domain robotics)
  (:objects
   can1 - Can
   can2 - Can
   target1 - Target
   target2 - Target
   target3 - Target
   robot_init_pose - Symbol
   gp_can1 - Symbol
   gp_can2 - Symbol
   pdp_target1 - Symbol
   pdp_target2 - Symbol
   pdp_target3 - Symbol
   pr2 - Robot
   )

  (:init
   (At can1 target1)
   (At can2 target2)
   (RobotAt pr2 robot_init_pose)
   (IsGP gp_can1 can1)
   (IsGP gp_can2 can2)
   (IsPDP pdp_target1 target1)
   (IsPDP pdp_target2 target2)
   (IsPDP pdp_target3 target3)
   )

  (:goal
   (and (At can1 target2) (At can2 target1))
   )
  )