; Auto-generated. Do not edit!


(cl:in-package tamp_ros-msg)


;//! \htmlinclude HLPlanResult.msg.html

(cl:defclass <HLPlanResult> (roslisp-msg-protocol:ros-message)
  ((steps
    :reader steps
    :initarg :steps
    :type (cl:vector tamp_ros-msg:MotionPlanResult)
   :initform (cl:make-array 0 :element-type 'tamp_ros-msg:MotionPlanResult :initial-element (cl:make-instance 'tamp_ros-msg:MotionPlanResult)))
   (path_to
    :reader path_to
    :initarg :path_to
    :type cl:string
    :initform "")
   (success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil)
   (cond
    :reader cond
    :initarg :cond
    :type cl:integer
    :initform 0))
)

(cl:defclass HLPlanResult (<HLPlanResult>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <HLPlanResult>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'HLPlanResult)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tamp_ros-msg:<HLPlanResult> is deprecated: use tamp_ros-msg:HLPlanResult instead.")))

(cl:ensure-generic-function 'steps-val :lambda-list '(m))
(cl:defmethod steps-val ((m <HLPlanResult>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tamp_ros-msg:steps-val is deprecated.  Use tamp_ros-msg:steps instead.")
  (steps m))

(cl:ensure-generic-function 'path_to-val :lambda-list '(m))
(cl:defmethod path_to-val ((m <HLPlanResult>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tamp_ros-msg:path_to-val is deprecated.  Use tamp_ros-msg:path_to instead.")
  (path_to m))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <HLPlanResult>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tamp_ros-msg:success-val is deprecated.  Use tamp_ros-msg:success instead.")
  (success m))

(cl:ensure-generic-function 'cond-val :lambda-list '(m))
(cl:defmethod cond-val ((m <HLPlanResult>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tamp_ros-msg:cond-val is deprecated.  Use tamp_ros-msg:cond instead.")
  (cond m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <HLPlanResult>) ostream)
  "Serializes a message object of type '<HLPlanResult>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'steps))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'steps))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'path_to))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'path_to))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
  (cl:let* ((signed (cl:slot-value msg 'cond)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <HLPlanResult>) istream)
  "Deserializes a message object of type '<HLPlanResult>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'steps) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'steps)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'tamp_ros-msg:MotionPlanResult))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'path_to) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'path_to) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'cond) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<HLPlanResult>)))
  "Returns string type for a message object of type '<HLPlanResult>"
  "tamp_ros/HLPlanResult")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'HLPlanResult)))
  "Returns string type for a message object of type 'HLPlanResult"
  "tamp_ros/HLPlanResult")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<HLPlanResult>)))
  "Returns md5sum for a message object of type '<HLPlanResult>"
  "3d15420a1ea98d1e6019df3adbc0e259")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'HLPlanResult)))
  "Returns md5sum for a message object of type 'HLPlanResult"
  "3d15420a1ea98d1e6019df3adbc0e259")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<HLPlanResult>)))
  "Returns full string definition for message of type '<HLPlanResult>"
  (cl:format cl:nil "MotionPlanResult[] steps~%string path_to~%bool success~%int32 cond~%~%================================================================================~%MSG: tamp_ros/MotionPlanResult~%std_msgs/Float32MultiArray[] traj~%string failed~%bool success~%int32 plan_id~%int32 cond~%string task~%~%================================================================================~%MSG: std_msgs/Float32MultiArray~%# Please look at the MultiArrayLayout message definition for~%# documentation on all multiarrays.~%~%MultiArrayLayout  layout        # specification of data layout~%float32[]         data          # array of data~%~%~%================================================================================~%MSG: std_msgs/MultiArrayLayout~%# The multiarray declares a generic multi-dimensional array of a~%# particular data type.  Dimensions are ordered from outer most~%# to inner most.~%~%MultiArrayDimension[] dim # Array of dimension properties~%uint32 data_offset        # padding elements at front of data~%~%# Accessors should ALWAYS be written in terms of dimension stride~%# and specified outer-most dimension first.~%# ~%# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]~%#~%# A standard, 3-channel 640x480 image with interleaved color channels~%# would be specified as:~%#~%# dim[0].label  = \"height\"~%# dim[0].size   = 480~%# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)~%# dim[1].label  = \"width\"~%# dim[1].size   = 640~%# dim[1].stride = 3*640 = 1920~%# dim[2].label  = \"channel\"~%# dim[2].size   = 3~%# dim[2].stride = 3~%#~%# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.~%~%================================================================================~%MSG: std_msgs/MultiArrayDimension~%string label   # label of given dimension~%uint32 size    # size of given dimension (in type units)~%uint32 stride  # stride of given dimension~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'HLPlanResult)))
  "Returns full string definition for message of type 'HLPlanResult"
  (cl:format cl:nil "MotionPlanResult[] steps~%string path_to~%bool success~%int32 cond~%~%================================================================================~%MSG: tamp_ros/MotionPlanResult~%std_msgs/Float32MultiArray[] traj~%string failed~%bool success~%int32 plan_id~%int32 cond~%string task~%~%================================================================================~%MSG: std_msgs/Float32MultiArray~%# Please look at the MultiArrayLayout message definition for~%# documentation on all multiarrays.~%~%MultiArrayLayout  layout        # specification of data layout~%float32[]         data          # array of data~%~%~%================================================================================~%MSG: std_msgs/MultiArrayLayout~%# The multiarray declares a generic multi-dimensional array of a~%# particular data type.  Dimensions are ordered from outer most~%# to inner most.~%~%MultiArrayDimension[] dim # Array of dimension properties~%uint32 data_offset        # padding elements at front of data~%~%# Accessors should ALWAYS be written in terms of dimension stride~%# and specified outer-most dimension first.~%# ~%# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]~%#~%# A standard, 3-channel 640x480 image with interleaved color channels~%# would be specified as:~%#~%# dim[0].label  = \"height\"~%# dim[0].size   = 480~%# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)~%# dim[1].label  = \"width\"~%# dim[1].size   = 640~%# dim[1].stride = 3*640 = 1920~%# dim[2].label  = \"channel\"~%# dim[2].size   = 3~%# dim[2].stride = 3~%#~%# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.~%~%================================================================================~%MSG: std_msgs/MultiArrayDimension~%string label   # label of given dimension~%uint32 size    # size of given dimension (in type units)~%uint32 stride  # stride of given dimension~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <HLPlanResult>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'steps) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
     4 (cl:length (cl:slot-value msg 'path_to))
     1
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <HLPlanResult>))
  "Converts a ROS message object to a list"
  (cl:list 'HLPlanResult
    (cl:cons ':steps (steps msg))
    (cl:cons ':path_to (path_to msg))
    (cl:cons ':success (success msg))
    (cl:cons ':cond (cond msg))
))
