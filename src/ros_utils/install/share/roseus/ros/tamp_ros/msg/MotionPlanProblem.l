;; Auto-generated. Do not edit!


(when (boundp 'tamp_ros::MotionPlanProblem)
  (if (not (find-package "TAMP_ROS"))
    (make-package "TAMP_ROS"))
  (shadow 'MotionPlanProblem (find-package "TAMP_ROS")))
(unless (find-package "TAMP_ROS::MOTIONPLANPROBLEM")
  (make-package "TAMP_ROS::MOTIONPLANPROBLEM"))

(in-package "ROS")
;;//! \htmlinclude MotionPlanProblem.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass tamp_ros::MotionPlanProblem
  :super ros::object
  :slots (_solver_id _prob_id _server_id _task _state _cond _traj_mean _use_prior _sigma _mu _logmass _mass _N _K _Do ))

(defmethod tamp_ros::MotionPlanProblem
  (:init
   (&key
    ((:solver_id __solver_id) 0)
    ((:prob_id __prob_id) 0)
    ((:server_id __server_id) 0)
    ((:task __task) "")
    ((:state __state) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:cond __cond) 0)
    ((:traj_mean __traj_mean) (let (r) (dotimes (i 0) (push (instance std_msgs::Float32MultiArray :init) r)) r))
    ((:use_prior __use_prior) nil)
    ((:sigma __sigma) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:mu __mu) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:logmass __logmass) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:mass __mass) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:N __N) 0)
    ((:K __K) 0)
    ((:Do __Do) 0)
    )
   (send-super :init)
   (setq _solver_id (round __solver_id))
   (setq _prob_id (round __prob_id))
   (setq _server_id (round __server_id))
   (setq _task (string __task))
   (setq _state __state)
   (setq _cond (round __cond))
   (setq _traj_mean __traj_mean)
   (setq _use_prior __use_prior)
   (setq _sigma __sigma)
   (setq _mu __mu)
   (setq _logmass __logmass)
   (setq _mass __mass)
   (setq _N (round __N))
   (setq _K (round __K))
   (setq _Do (round __Do))
   self)
  (:solver_id
   (&optional __solver_id)
   (if __solver_id (setq _solver_id __solver_id)) _solver_id)
  (:prob_id
   (&optional __prob_id)
   (if __prob_id (setq _prob_id __prob_id)) _prob_id)
  (:server_id
   (&optional __server_id)
   (if __server_id (setq _server_id __server_id)) _server_id)
  (:task
   (&optional __task)
   (if __task (setq _task __task)) _task)
  (:state
   (&optional __state)
   (if __state (setq _state __state)) _state)
  (:cond
   (&optional __cond)
   (if __cond (setq _cond __cond)) _cond)
  (:traj_mean
   (&rest __traj_mean)
   (if (keywordp (car __traj_mean))
       (send* _traj_mean __traj_mean)
     (progn
       (if __traj_mean (setq _traj_mean (car __traj_mean)))
       _traj_mean)))
  (:use_prior
   (&optional __use_prior)
   (if __use_prior (setq _use_prior __use_prior)) _use_prior)
  (:sigma
   (&optional __sigma)
   (if __sigma (setq _sigma __sigma)) _sigma)
  (:mu
   (&optional __mu)
   (if __mu (setq _mu __mu)) _mu)
  (:logmass
   (&optional __logmass)
   (if __logmass (setq _logmass __logmass)) _logmass)
  (:mass
   (&optional __mass)
   (if __mass (setq _mass __mass)) _mass)
  (:N
   (&optional __N)
   (if __N (setq _N __N)) _N)
  (:K
   (&optional __K)
   (if __K (setq _K __K)) _K)
  (:Do
   (&optional __Do)
   (if __Do (setq _Do __Do)) _Do)
  (:serialization-length
   ()
   (+
    ;; int32 _solver_id
    4
    ;; int32 _prob_id
    4
    ;; int32 _server_id
    4
    ;; string _task
    4 (length _task)
    ;; float32[] _state
    (* 4    (length _state)) 4
    ;; int32 _cond
    4
    ;; std_msgs/Float32MultiArray[] _traj_mean
    (apply #'+ (send-all _traj_mean :serialization-length)) 4
    ;; bool _use_prior
    1
    ;; float32[] _sigma
    (* 4    (length _sigma)) 4
    ;; float32[] _mu
    (* 4    (length _mu)) 4
    ;; float32[] _logmass
    (* 4    (length _logmass)) 4
    ;; float32[] _mass
    (* 4    (length _mass)) 4
    ;; int32 _N
    4
    ;; int32 _K
    4
    ;; int32 _Do
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int32 _solver_id
       (write-long _solver_id s)
     ;; int32 _prob_id
       (write-long _prob_id s)
     ;; int32 _server_id
       (write-long _server_id s)
     ;; string _task
       (write-long (length _task) s) (princ _task s)
     ;; float32[] _state
     (write-long (length _state) s)
     (dotimes (i (length _state))
       (sys::poke (elt _state i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; int32 _cond
       (write-long _cond s)
     ;; std_msgs/Float32MultiArray[] _traj_mean
     (write-long (length _traj_mean) s)
     (dolist (elem _traj_mean)
       (send elem :serialize s)
       )
     ;; bool _use_prior
       (if _use_prior (write-byte -1 s) (write-byte 0 s))
     ;; float32[] _sigma
     (write-long (length _sigma) s)
     (dotimes (i (length _sigma))
       (sys::poke (elt _sigma i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32[] _mu
     (write-long (length _mu) s)
     (dotimes (i (length _mu))
       (sys::poke (elt _mu i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32[] _logmass
     (write-long (length _logmass) s)
     (dotimes (i (length _logmass))
       (sys::poke (elt _logmass i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32[] _mass
     (write-long (length _mass) s)
     (dotimes (i (length _mass))
       (sys::poke (elt _mass i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; int32 _N
       (write-long _N s)
     ;; int32 _K
       (write-long _K s)
     ;; int32 _Do
       (write-long _Do s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int32 _solver_id
     (setq _solver_id (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _prob_id
     (setq _prob_id (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _server_id
     (setq _server_id (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; string _task
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _task (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; float32[] _state
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _state (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _state i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;; int32 _cond
     (setq _cond (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; std_msgs/Float32MultiArray[] _traj_mean
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _traj_mean (let (r) (dotimes (i n) (push (instance std_msgs::Float32MultiArray :init) r)) r))
     (dolist (elem- _traj_mean)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; bool _use_prior
     (setq _use_prior (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; float32[] _sigma
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _sigma (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _sigma i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;; float32[] _mu
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _mu (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _mu i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;; float32[] _logmass
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _logmass (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _logmass i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;; float32[] _mass
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _mass (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _mass i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;; int32 _N
     (setq _N (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _K
     (setq _K (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _Do
     (setq _Do (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get tamp_ros::MotionPlanProblem :md5sum-) "62f92843a78550529c22bfbaad9b886c")
(setf (get tamp_ros::MotionPlanProblem :datatype-) "tamp_ros/MotionPlanProblem")
(setf (get tamp_ros::MotionPlanProblem :definition-)
      "int32 solver_id
int32 prob_id
int32 server_id
string task
float32[] state
int32 cond
std_msgs/Float32MultiArray[] traj_mean

bool use_prior
float32[] sigma
float32[] mu
float32[] logmass
float32[] mass
int32 N
int32 K
int32 Do

================================================================================
MSG: std_msgs/Float32MultiArray
# Please look at the MultiArrayLayout message definition for
# documentation on all multiarrays.

MultiArrayLayout  layout        # specification of data layout
float32[]         data          # array of data


================================================================================
MSG: std_msgs/MultiArrayLayout
# The multiarray declares a generic multi-dimensional array of a
# particular data type.  Dimensions are ordered from outer most
# to inner most.

MultiArrayDimension[] dim # Array of dimension properties
uint32 data_offset        # padding elements at front of data

# Accessors should ALWAYS be written in terms of dimension stride
# and specified outer-most dimension first.
# 
# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]
#
# A standard, 3-channel 640x480 image with interleaved color channels
# would be specified as:
#
# dim[0].label  = \"height\"
# dim[0].size   = 480
# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)
# dim[1].label  = \"width\"
# dim[1].size   = 640
# dim[1].stride = 3*640 = 1920
# dim[2].label  = \"channel\"
# dim[2].size   = 3
# dim[2].stride = 3
#
# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.

================================================================================
MSG: std_msgs/MultiArrayDimension
string label   # label of given dimension
uint32 size    # size of given dimension (in type units)
uint32 stride  # stride of given dimension
")



(provide :tamp_ros/MotionPlanProblem "62f92843a78550529c22bfbaad9b886c")


