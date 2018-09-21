;; Auto-generated. Do not edit!


(when (boundp 'tamp_ros::PolicyProb)
  (if (not (find-package "TAMP_ROS"))
    (make-package "TAMP_ROS"))
  (shadow 'PolicyProb (find-package "TAMP_ROS")))
(unless (find-package "TAMP_ROS::POLICYPROB")
  (make-package "TAMP_ROS::POLICYPROB"))
(unless (find-package "TAMP_ROS::POLICYPROBREQUEST")
  (make-package "TAMP_ROS::POLICYPROBREQUEST"))
(unless (find-package "TAMP_ROS::POLICYPROBRESPONSE")
  (make-package "TAMP_ROS::POLICYPROBRESPONSE"))

(in-package "ROS")

(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass tamp_ros::PolicyProbRequest
  :super ros::object
  :slots (_obs _task ))

(defmethod tamp_ros::PolicyProbRequest
  (:init
   (&key
    ((:obs __obs) (let (r) (dotimes (i 0) (push (instance std_msgs::Float32MultiArray :init) r)) r))
    ((:task __task) "")
    )
   (send-super :init)
   (setq _obs __obs)
   (setq _task (string __task))
   self)
  (:obs
   (&rest __obs)
   (if (keywordp (car __obs))
       (send* _obs __obs)
     (progn
       (if __obs (setq _obs (car __obs)))
       _obs)))
  (:task
   (&optional __task)
   (if __task (setq _task __task)) _task)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Float32MultiArray[] _obs
    (apply #'+ (send-all _obs :serialization-length)) 4
    ;; string _task
    4 (length _task)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Float32MultiArray[] _obs
     (write-long (length _obs) s)
     (dolist (elem _obs)
       (send elem :serialize s)
       )
     ;; string _task
       (write-long (length _task) s) (princ _task s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Float32MultiArray[] _obs
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _obs (let (r) (dotimes (i n) (push (instance std_msgs::Float32MultiArray :init) r)) r))
     (dolist (elem- _obs)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; string _task
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _task (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass tamp_ros::PolicyProbResponse
  :super ros::object
  :slots (_mu _sigma ))

(defmethod tamp_ros::PolicyProbResponse
  (:init
   (&key
    ((:mu __mu) (let (r) (dotimes (i 0) (push (instance std_msgs::Float32MultiArray :init) r)) r))
    ((:sigma __sigma) (let (r) (dotimes (i 0) (push (instance std_msgs::Float32MultiArray :init) r)) r))
    )
   (send-super :init)
   (setq _mu __mu)
   (setq _sigma __sigma)
   self)
  (:mu
   (&rest __mu)
   (if (keywordp (car __mu))
       (send* _mu __mu)
     (progn
       (if __mu (setq _mu (car __mu)))
       _mu)))
  (:sigma
   (&rest __sigma)
   (if (keywordp (car __sigma))
       (send* _sigma __sigma)
     (progn
       (if __sigma (setq _sigma (car __sigma)))
       _sigma)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Float32MultiArray[] _mu
    (apply #'+ (send-all _mu :serialization-length)) 4
    ;; std_msgs/Float32MultiArray[] _sigma
    (apply #'+ (send-all _sigma :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Float32MultiArray[] _mu
     (write-long (length _mu) s)
     (dolist (elem _mu)
       (send elem :serialize s)
       )
     ;; std_msgs/Float32MultiArray[] _sigma
     (write-long (length _sigma) s)
     (dolist (elem _sigma)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Float32MultiArray[] _mu
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _mu (let (r) (dotimes (i n) (push (instance std_msgs::Float32MultiArray :init) r)) r))
     (dolist (elem- _mu)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; std_msgs/Float32MultiArray[] _sigma
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _sigma (let (r) (dotimes (i n) (push (instance std_msgs::Float32MultiArray :init) r)) r))
     (dolist (elem- _sigma)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(defclass tamp_ros::PolicyProb
  :super ros::object
  :slots ())

(setf (get tamp_ros::PolicyProb :md5sum-) "543016ad28d3afef79460f66829d896a")
(setf (get tamp_ros::PolicyProb :datatype-) "tamp_ros/PolicyProb")
(setf (get tamp_ros::PolicyProb :request) tamp_ros::PolicyProbRequest)
(setf (get tamp_ros::PolicyProb :response) tamp_ros::PolicyProbResponse)

(defmethod tamp_ros::PolicyProbRequest
  (:response () (instance tamp_ros::PolicyProbResponse :init)))

(setf (get tamp_ros::PolicyProbRequest :md5sum-) "543016ad28d3afef79460f66829d896a")
(setf (get tamp_ros::PolicyProbRequest :datatype-) "tamp_ros/PolicyProbRequest")
(setf (get tamp_ros::PolicyProbRequest :definition-)
      "std_msgs/Float32MultiArray[] obs
string task


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

std_msgs/Float32MultiArray[] mu
std_msgs/Float32MultiArray[] sigma


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

(setf (get tamp_ros::PolicyProbResponse :md5sum-) "543016ad28d3afef79460f66829d896a")
(setf (get tamp_ros::PolicyProbResponse :datatype-) "tamp_ros/PolicyProbResponse")
(setf (get tamp_ros::PolicyProbResponse :definition-)
      "std_msgs/Float32MultiArray[] obs
string task


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

std_msgs/Float32MultiArray[] mu
std_msgs/Float32MultiArray[] sigma


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



(provide :tamp_ros/PolicyProb "543016ad28d3afef79460f66829d896a")


