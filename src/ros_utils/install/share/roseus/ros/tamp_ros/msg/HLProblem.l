;; Auto-generated. Do not edit!


(when (boundp 'tamp_ros::HLProblem)
  (if (not (find-package "TAMP_ROS"))
    (make-package "TAMP_ROS"))
  (shadow 'HLProblem (find-package "TAMP_ROS")))
(unless (find-package "TAMP_ROS::HLPROBLEM")
  (make-package "TAMP_ROS::HLPROBLEM"))

(in-package "ROS")
;;//! \htmlinclude HLProblem.msg.html


(defclass tamp_ros::HLProblem
  :super ros::object
  :slots (_solver_id _server_id _init_state _cond _path_to ))

(defmethod tamp_ros::HLProblem
  (:init
   (&key
    ((:solver_id __solver_id) 0)
    ((:server_id __server_id) 0)
    ((:init_state __init_state) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:cond __cond) 0)
    ((:path_to __path_to) "")
    )
   (send-super :init)
   (setq _solver_id (round __solver_id))
   (setq _server_id (round __server_id))
   (setq _init_state __init_state)
   (setq _cond (round __cond))
   (setq _path_to (string __path_to))
   self)
  (:solver_id
   (&optional __solver_id)
   (if __solver_id (setq _solver_id __solver_id)) _solver_id)
  (:server_id
   (&optional __server_id)
   (if __server_id (setq _server_id __server_id)) _server_id)
  (:init_state
   (&optional __init_state)
   (if __init_state (setq _init_state __init_state)) _init_state)
  (:cond
   (&optional __cond)
   (if __cond (setq _cond __cond)) _cond)
  (:path_to
   (&optional __path_to)
   (if __path_to (setq _path_to __path_to)) _path_to)
  (:serialization-length
   ()
   (+
    ;; int32 _solver_id
    4
    ;; int32 _server_id
    4
    ;; float32[] _init_state
    (* 4    (length _init_state)) 4
    ;; int32 _cond
    4
    ;; string _path_to
    4 (length _path_to)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int32 _solver_id
       (write-long _solver_id s)
     ;; int32 _server_id
       (write-long _server_id s)
     ;; float32[] _init_state
     (write-long (length _init_state) s)
     (dotimes (i (length _init_state))
       (sys::poke (elt _init_state i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; int32 _cond
       (write-long _cond s)
     ;; string _path_to
       (write-long (length _path_to) s) (princ _path_to s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int32 _solver_id
     (setq _solver_id (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _server_id
     (setq _server_id (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; float32[] _init_state
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _init_state (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _init_state i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;; int32 _cond
     (setq _cond (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; string _path_to
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _path_to (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(setf (get tamp_ros::HLProblem :md5sum-) "1a2adc264904a8c2fffd546791b7a9ec")
(setf (get tamp_ros::HLProblem :datatype-) "tamp_ros/HLProblem")
(setf (get tamp_ros::HLProblem :definition-)
      "int32 solver_id
int32 server_id
float32[] init_state
int32 cond
string path_to

")



(provide :tamp_ros/HLProblem "1a2adc264904a8c2fffd546791b7a9ec")


