;; Auto-generated. Do not edit!


(when (boundp 'tamp_ros::MotionPlan)
  (if (not (find-package "TAMP_ROS"))
    (make-package "TAMP_ROS"))
  (shadow 'MotionPlan (find-package "TAMP_ROS")))
(unless (find-package "TAMP_ROS::MOTIONPLAN")
  (make-package "TAMP_ROS::MOTIONPLAN"))
(unless (find-package "TAMP_ROS::MOTIONPLANREQUEST")
  (make-package "TAMP_ROS::MOTIONPLANREQUEST"))
(unless (find-package "TAMP_ROS::MOTIONPLANRESPONSE")
  (make-package "TAMP_ROS::MOTIONPLANRESPONSE"))

(in-package "ROS")

(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass tamp_ros::MotionPlanRequest
  :super ros::object
  :slots (_state _task _obj _targ _condition _traj_mean ))

(defmethod tamp_ros::MotionPlanRequest
  (:init
   (&key
    ((:state __state) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:task __task) "")
    ((:obj __obj) "")
    ((:targ __targ) "")
    ((:condition __condition) 0)
    ((:traj_mean __traj_mean) (let (r) (dotimes (i 0) (push (instance std_msgs::Float32MultiArray :init) r)) r))
    )
   (send-super :init)
   (setq _state __state)
   (setq _task (string __task))
   (setq _obj (string __obj))
   (setq _targ (string __targ))
   (setq _condition (round __condition))
   (setq _traj_mean __traj_mean)
   self)
  (:state
   (&optional __state)
   (if __state (setq _state __state)) _state)
  (:task
   (&optional __task)
   (if __task (setq _task __task)) _task)
  (:obj
   (&optional __obj)
   (if __obj (setq _obj __obj)) _obj)
  (:targ
   (&optional __targ)
   (if __targ (setq _targ __targ)) _targ)
  (:condition
   (&optional __condition)
   (if __condition (setq _condition __condition)) _condition)
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
    ;; float32[] _state
    (* 4    (length _state)) 4
    ;; string _task
    4 (length _task)
    ;; string _obj
    4 (length _obj)
    ;; string _targ
    4 (length _targ)
    ;; int32 _condition
    4
    ;; std_msgs/Float32MultiArray[] _traj_mean
    (apply #'+ (send-all _traj_mean :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32[] _state
     (write-long (length _state) s)
     (dotimes (i (length _state))
       (sys::poke (elt _state i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; string _task
       (write-long (length _task) s) (princ _task s)
     ;; string _obj
       (write-long (length _obj) s) (princ _obj s)
     ;; string _targ
       (write-long (length _targ) s) (princ _targ s)
     ;; int32 _condition
       (write-long _condition s)
     ;; std_msgs/Float32MultiArray[] _traj_mean
     (write-long (length _traj_mean) s)
     (dolist (elem _traj_mean)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32[] _state
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _state (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _state i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;; string _task
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _task (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _obj
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _obj (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _targ
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _targ (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; int32 _condition
     (setq _condition (sys::peek buf ptr- :integer)) (incf ptr- 4)
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

(defclass tamp_ros::MotionPlanResponse
  :super ros::object
  :slots (_traj _failed _succes ))

(defmethod tamp_ros::MotionPlanResponse
  (:init
   (&key
    ((:traj __traj) (let (r) (dotimes (i 0) (push (instance std_msgs::Float32MultiArray :init) r)) r))
    ((:failed __failed) "")
    ((:succes __succes) nil)
    )
   (send-super :init)
   (setq _traj __traj)
   (setq _failed (string __failed))
   (setq _succes __succes)
   self)
  (:traj
   (&rest __traj)
   (if (keywordp (car __traj))
       (send* _traj __traj)
     (progn
       (if __traj (setq _traj (car __traj)))
       _traj)))
  (:failed
   (&optional __failed)
   (if __failed (setq _failed __failed)) _failed)
  (:succes
   (&optional __succes)
   (if __succes (setq _succes __succes)) _succes)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Float32MultiArray[] _traj
    (apply #'+ (send-all _traj :serialization-length)) 4
    ;; string _failed
    4 (length _failed)
    ;; bool _succes
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Float32MultiArray[] _traj
     (write-long (length _traj) s)
     (dolist (elem _traj)
       (send elem :serialize s)
       )
     ;; string _failed
       (write-long (length _failed) s) (princ _failed s)
     ;; bool _succes
       (if _succes (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Float32MultiArray[] _traj
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _traj (let (r) (dotimes (i n) (push (instance std_msgs::Float32MultiArray :init) r)) r))
     (dolist (elem- _traj)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; string _failed
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _failed (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; bool _succes
     (setq _succes (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass tamp_ros::MotionPlan
  :super ros::object
  :slots ())

(setf (get tamp_ros::MotionPlan :md5sum-) "3979cf2783d6a8eedc4996bbfe7b87c9")
(setf (get tamp_ros::MotionPlan :datatype-) "tamp_ros/MotionPlan")
(setf (get tamp_ros::MotionPlan :request) tamp_ros::MotionPlanRequest)
(setf (get tamp_ros::MotionPlan :response) tamp_ros::MotionPlanResponse)

(defmethod tamp_ros::MotionPlanRequest
  (:response () (instance tamp_ros::MotionPlanResponse :init)))

(setf (get tamp_ros::MotionPlanRequest :md5sum-) "3979cf2783d6a8eedc4996bbfe7b87c9")
(setf (get tamp_ros::MotionPlanRequest :datatype-) "tamp_ros/MotionPlanRequest")
(setf (get tamp_ros::MotionPlanRequest :definition-)
      "float32[] state
string task
string obj
string targ
int32 condition
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
---

std_msgs/Float32MultiArray[] traj
string failed
bool succes


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
")

(setf (get tamp_ros::MotionPlanResponse :md5sum-) "3979cf2783d6a8eedc4996bbfe7b87c9")
(setf (get tamp_ros::MotionPlanResponse :datatype-) "tamp_ros/MotionPlanResponse")
(setf (get tamp_ros::MotionPlanResponse :definition-)
      "float32[] state
string task
string obj
string targ
int32 condition
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
---

std_msgs/Float32MultiArray[] traj
string failed
bool succes


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
")



(provide :tamp_ros/MotionPlan "3979cf2783d6a8eedc4996bbfe7b87c9")


