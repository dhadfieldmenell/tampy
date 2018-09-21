;; Auto-generated. Do not edit!


(when (boundp 'tamp_ros::PlanProb)
  (if (not (find-package "TAMP_ROS"))
    (make-package "TAMP_ROS"))
  (shadow 'PlanProb (find-package "TAMP_ROS")))
(unless (find-package "TAMP_ROS::PLANPROB")
  (make-package "TAMP_ROS::PLANPROB"))

(in-package "ROS")
;;//! \htmlinclude PlanProb.msg.html


(defclass tamp_ros::PlanProb
  :super ros::object
  :slots (_prob_id _task _object _target _state ))

(defmethod tamp_ros::PlanProb
  (:init
   (&key
    ((:prob_id __prob_id) 0)
    ((:task __task) "")
    ((:object __object) "")
    ((:target __target) "")
    ((:state __state) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _prob_id (round __prob_id))
   (setq _task (string __task))
   (setq _object (string __object))
   (setq _target (string __target))
   (setq _state __state)
   self)
  (:prob_id
   (&optional __prob_id)
   (if __prob_id (setq _prob_id __prob_id)) _prob_id)
  (:task
   (&optional __task)
   (if __task (setq _task __task)) _task)
  (:object
   (&optional __object)
   (if __object (setq _object __object)) _object)
  (:target
   (&optional __target)
   (if __target (setq _target __target)) _target)
  (:state
   (&optional __state)
   (if __state (setq _state __state)) _state)
  (:serialization-length
   ()
   (+
    ;; int32 _prob_id
    4
    ;; string _task
    4 (length _task)
    ;; string _object
    4 (length _object)
    ;; string _target
    4 (length _target)
    ;; float32[] _state
    (* 4    (length _state)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int32 _prob_id
       (write-long _prob_id s)
     ;; string _task
       (write-long (length _task) s) (princ _task s)
     ;; string _object
       (write-long (length _object) s) (princ _object s)
     ;; string _target
       (write-long (length _target) s) (princ _target s)
     ;; float32[] _state
     (write-long (length _state) s)
     (dotimes (i (length _state))
       (sys::poke (elt _state i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int32 _prob_id
     (setq _prob_id (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; string _task
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _task (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _object
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _object (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _target
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _target (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; float32[] _state
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _state (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _state i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;;
   self)
  )

(setf (get tamp_ros::PlanProb :md5sum-) "bacbc44c2a384d436608cc453c774b3b")
(setf (get tamp_ros::PlanProb :datatype-) "tamp_ros/PlanProb")
(setf (get tamp_ros::PlanProb :definition-)
      "int32 prob_id
string task
string object
string target
float32[] state

")



(provide :tamp_ros/PlanProb "bacbc44c2a384d436608cc453c774b3b")


