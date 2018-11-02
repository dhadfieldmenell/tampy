;; Auto-generated. Do not edit!


(when (boundp 'tamp_ros::HLPlanResult)
  (if (not (find-package "TAMP_ROS"))
    (make-package "TAMP_ROS"))
  (shadow 'HLPlanResult (find-package "TAMP_ROS")))
(unless (find-package "TAMP_ROS::HLPLANRESULT")
  (make-package "TAMP_ROS::HLPLANRESULT"))

(in-package "ROS")
;;//! \htmlinclude HLPlanResult.msg.html


(defclass tamp_ros::HLPlanResult
  :super ros::object
  :slots (_steps _path_to _success _cond ))

(defmethod tamp_ros::HLPlanResult
  (:init
   (&key
    ((:steps __steps) (let (r) (dotimes (i 0) (push (instance tamp_ros::MotionPlanResult :init) r)) r))
    ((:path_to __path_to) "")
    ((:success __success) nil)
    ((:cond __cond) 0)
    )
   (send-super :init)
   (setq _steps __steps)
   (setq _path_to (string __path_to))
   (setq _success __success)
   (setq _cond (round __cond))
   self)
  (:steps
   (&rest __steps)
   (if (keywordp (car __steps))
       (send* _steps __steps)
     (progn
       (if __steps (setq _steps (car __steps)))
       _steps)))
  (:path_to
   (&optional __path_to)
   (if __path_to (setq _path_to __path_to)) _path_to)
  (:success
   (&optional __success)
   (if __success (setq _success __success)) _success)
  (:cond
   (&optional __cond)
   (if __cond (setq _cond __cond)) _cond)
  (:serialization-length
   ()
   (+
    ;; tamp_ros/MotionPlanResult[] _steps
    (apply #'+ (send-all _steps :serialization-length)) 4
    ;; string _path_to
    4 (length _path_to)
    ;; bool _success
    1
    ;; int32 _cond
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; tamp_ros/MotionPlanResult[] _steps
     (write-long (length _steps) s)
     (dolist (elem _steps)
       (send elem :serialize s)
       )
     ;; string _path_to
       (write-long (length _path_to) s) (princ _path_to s)
     ;; bool _success
       (if _success (write-byte -1 s) (write-byte 0 s))
     ;; int32 _cond
       (write-long _cond s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; tamp_ros/MotionPlanResult[] _steps
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _steps (let (r) (dotimes (i n) (push (instance tamp_ros::MotionPlanResult :init) r)) r))
     (dolist (elem- _steps)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; string _path_to
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _path_to (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; bool _success
     (setq _success (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; int32 _cond
     (setq _cond (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get tamp_ros::HLPlanResult :md5sum-) "19bff39c2204ab093accc09544e93f76")
(setf (get tamp_ros::HLPlanResult :datatype-) "tamp_ros/HLPlanResult")
(setf (get tamp_ros::HLPlanResult :definition-)
      "MotionPlanResult[] steps
string path_to
bool success
int32 cond

================================================================================
MSG: tamp_ros/MotionPlanResult
std_msgs/Float32MultiArray[] traj
string failed
bool success
int32 plan_id
int32 cond
string task
string obj
string targ

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



(provide :tamp_ros/HLPlanResult "19bff39c2204ab093accc09544e93f76")


