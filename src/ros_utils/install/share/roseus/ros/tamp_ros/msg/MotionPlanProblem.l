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
  :slots (_prob_id _task _obj _targ _state _cond _traj_mean ))

(defmethod tamp_ros::MotionPlanProblem
  (:init
   (&key
    ((:prob_id __prob_id) 0)
    ((:task __task) "")
    ((:obj __obj) "")
    ((:targ __targ) "")
    ((:state __state) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:cond __cond) 0)
    ((:traj_mean __traj_mean) (let (r) (dotimes (i 0) (push (instance std_msgs::Float32MultiArray :init) r)) r))
    )
   (send-super :init)
   (setq _prob_id (round __prob_id))
   (setq _task (string __task))
   (setq _obj (string __obj))
   (setq _targ (string __targ))
   (setq _state __state)
   (setq _cond (round __cond))
   (setq _traj_mean __traj_mean)
   self)
  (:prob_id
   (&optional __prob_id)
   (if __prob_id (setq _prob_id __prob_id)) _prob_id)
  (:task
   (&optional __task)
   (if __task (setq _task __task)) _task)
  (:obj
   (&optional __obj)
   (if __obj (setq _obj __obj)) _obj)
  (:targ
   (&optional __targ)
   (if __targ (setq _targ __targ)) _targ)
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
  (:serialization-length
   ()
   (+
    ;; int32 _prob_id
    4
    ;; string _task
    4 (length _task)
    ;; string _obj
    4 (length _obj)
    ;; string _targ
    4 (length _targ)
    ;; float32[] _state
    (* 4    (length _state)) 4
    ;; int32 _cond
    4
    ;; std_msgs/Float32MultiArray[] _traj_mean
    (apply #'+ (send-all _traj_mean :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int32 _prob_id
       (write-long _prob_id s)
     ;; string _task
       (write-long (length _task) s) (princ _task s)
     ;; string _obj
       (write-long (length _obj) s) (princ _obj s)
     ;; string _targ
       (write-long (length _targ) s) (princ _targ s)
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
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int32 _prob_id
     (setq _prob_id (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; string _task
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _task (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _obj
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _obj (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _targ
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _targ (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
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
   ;;
   self)
  )

(setf (get tamp_ros::MotionPlanProblem :md5sum-) "fd42f7570f643f3f3f8d1c5e9bc3c462")
(setf (get tamp_ros::MotionPlanProblem :datatype-) "tamp_ros/MotionPlanProblem")
(setf (get tamp_ros::MotionPlanProblem :definition-)
      "int32 prob_id
string task
string obj
string targ
float32[] state
int32 cond
std_msgs/Float32MultiArray[] traj_mean

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



(provide :tamp_ros/MotionPlanProblem "fd42f7570f643f3f3f8d1c5e9bc3c462")


