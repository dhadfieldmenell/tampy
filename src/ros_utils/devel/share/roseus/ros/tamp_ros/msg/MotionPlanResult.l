;; Auto-generated. Do not edit!


(when (boundp 'tamp_ros::MotionPlanResult)
  (if (not (find-package "TAMP_ROS"))
    (make-package "TAMP_ROS"))
  (shadow 'MotionPlanResult (find-package "TAMP_ROS")))
(unless (find-package "TAMP_ROS::MOTIONPLANRESULT")
  (make-package "TAMP_ROS::MOTIONPLANRESULT"))

(in-package "ROS")
;;//! \htmlinclude MotionPlanResult.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass tamp_ros::MotionPlanResult
  :super ros::object
  :slots (_traj _failed _success _plan_id _cond _task ))

(defmethod tamp_ros::MotionPlanResult
  (:init
   (&key
    ((:traj __traj) (let (r) (dotimes (i 0) (push (instance std_msgs::Float32MultiArray :init) r)) r))
    ((:failed __failed) "")
    ((:success __success) nil)
    ((:plan_id __plan_id) 0)
    ((:cond __cond) 0)
    ((:task __task) "")
    )
   (send-super :init)
   (setq _traj __traj)
   (setq _failed (string __failed))
   (setq _success __success)
   (setq _plan_id (round __plan_id))
   (setq _cond (round __cond))
   (setq _task (string __task))
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
  (:success
   (&optional __success)
   (if __success (setq _success __success)) _success)
  (:plan_id
   (&optional __plan_id)
   (if __plan_id (setq _plan_id __plan_id)) _plan_id)
  (:cond
   (&optional __cond)
   (if __cond (setq _cond __cond)) _cond)
  (:task
   (&optional __task)
   (if __task (setq _task __task)) _task)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Float32MultiArray[] _traj
    (apply #'+ (send-all _traj :serialization-length)) 4
    ;; string _failed
    4 (length _failed)
    ;; bool _success
    1
    ;; int32 _plan_id
    4
    ;; int32 _cond
    4
    ;; string _task
    4 (length _task)
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
     ;; bool _success
       (if _success (write-byte -1 s) (write-byte 0 s))
     ;; int32 _plan_id
       (write-long _plan_id s)
     ;; int32 _cond
       (write-long _cond s)
     ;; string _task
       (write-long (length _task) s) (princ _task s)
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
   ;; bool _success
     (setq _success (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; int32 _plan_id
     (setq _plan_id (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _cond
     (setq _cond (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; string _task
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _task (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(setf (get tamp_ros::MotionPlanResult :md5sum-) "5efc67c8ca4a7fce7ecdcac89381d056")
(setf (get tamp_ros::MotionPlanResult :datatype-) "tamp_ros/MotionPlanResult")
(setf (get tamp_ros::MotionPlanResult :definition-)
      "std_msgs/Float32MultiArray[] traj
string failed
bool success
int32 plan_id
int32 cond
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
uint32 stride  # stride of given dimension
")



(provide :tamp_ros/MotionPlanResult "5efc67c8ca4a7fce7ecdcac89381d056")


