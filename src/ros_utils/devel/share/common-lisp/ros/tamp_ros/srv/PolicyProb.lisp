; Auto-generated. Do not edit!


(cl:in-package tamp_ros-srv)


;//! \htmlinclude PolicyProb-request.msg.html

(cl:defclass <PolicyProb-request> (roslisp-msg-protocol:ros-message)
  ((obs
    :reader obs
    :initarg :obs
    :type (cl:vector std_msgs-msg:Float32MultiArray)
   :initform (cl:make-array 0 :element-type 'std_msgs-msg:Float32MultiArray :initial-element (cl:make-instance 'std_msgs-msg:Float32MultiArray)))
   (task
    :reader task
    :initarg :task
    :type cl:string
    :initform ""))
)

(cl:defclass PolicyProb-request (<PolicyProb-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <PolicyProb-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'PolicyProb-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tamp_ros-srv:<PolicyProb-request> is deprecated: use tamp_ros-srv:PolicyProb-request instead.")))

(cl:ensure-generic-function 'obs-val :lambda-list '(m))
(cl:defmethod obs-val ((m <PolicyProb-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tamp_ros-srv:obs-val is deprecated.  Use tamp_ros-srv:obs instead.")
  (obs m))

(cl:ensure-generic-function 'task-val :lambda-list '(m))
(cl:defmethod task-val ((m <PolicyProb-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tamp_ros-srv:task-val is deprecated.  Use tamp_ros-srv:task instead.")
  (task m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <PolicyProb-request>) ostream)
  "Serializes a message object of type '<PolicyProb-request>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'obs))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'obs))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'task))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'task))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <PolicyProb-request>) istream)
  "Deserializes a message object of type '<PolicyProb-request>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'obs) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'obs)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'std_msgs-msg:Float32MultiArray))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'task) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'task) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<PolicyProb-request>)))
  "Returns string type for a service object of type '<PolicyProb-request>"
  "tamp_ros/PolicyProbRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'PolicyProb-request)))
  "Returns string type for a service object of type 'PolicyProb-request"
  "tamp_ros/PolicyProbRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<PolicyProb-request>)))
  "Returns md5sum for a message object of type '<PolicyProb-request>"
  "543016ad28d3afef79460f66829d896a")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'PolicyProb-request)))
  "Returns md5sum for a message object of type 'PolicyProb-request"
  "543016ad28d3afef79460f66829d896a")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<PolicyProb-request>)))
  "Returns full string definition for message of type '<PolicyProb-request>"
  (cl:format cl:nil "std_msgs/Float32MultiArray[] obs~%string task~%~%~%================================================================================~%MSG: std_msgs/Float32MultiArray~%# Please look at the MultiArrayLayout message definition for~%# documentation on all multiarrays.~%~%MultiArrayLayout  layout        # specification of data layout~%float32[]         data          # array of data~%~%~%================================================================================~%MSG: std_msgs/MultiArrayLayout~%# The multiarray declares a generic multi-dimensional array of a~%# particular data type.  Dimensions are ordered from outer most~%# to inner most.~%~%MultiArrayDimension[] dim # Array of dimension properties~%uint32 data_offset        # padding elements at front of data~%~%# Accessors should ALWAYS be written in terms of dimension stride~%# and specified outer-most dimension first.~%# ~%# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]~%#~%# A standard, 3-channel 640x480 image with interleaved color channels~%# would be specified as:~%#~%# dim[0].label  = \"height\"~%# dim[0].size   = 480~%# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)~%# dim[1].label  = \"width\"~%# dim[1].size   = 640~%# dim[1].stride = 3*640 = 1920~%# dim[2].label  = \"channel\"~%# dim[2].size   = 3~%# dim[2].stride = 3~%#~%# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.~%~%================================================================================~%MSG: std_msgs/MultiArrayDimension~%string label   # label of given dimension~%uint32 size    # size of given dimension (in type units)~%uint32 stride  # stride of given dimension~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'PolicyProb-request)))
  "Returns full string definition for message of type 'PolicyProb-request"
  (cl:format cl:nil "std_msgs/Float32MultiArray[] obs~%string task~%~%~%================================================================================~%MSG: std_msgs/Float32MultiArray~%# Please look at the MultiArrayLayout message definition for~%# documentation on all multiarrays.~%~%MultiArrayLayout  layout        # specification of data layout~%float32[]         data          # array of data~%~%~%================================================================================~%MSG: std_msgs/MultiArrayLayout~%# The multiarray declares a generic multi-dimensional array of a~%# particular data type.  Dimensions are ordered from outer most~%# to inner most.~%~%MultiArrayDimension[] dim # Array of dimension properties~%uint32 data_offset        # padding elements at front of data~%~%# Accessors should ALWAYS be written in terms of dimension stride~%# and specified outer-most dimension first.~%# ~%# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]~%#~%# A standard, 3-channel 640x480 image with interleaved color channels~%# would be specified as:~%#~%# dim[0].label  = \"height\"~%# dim[0].size   = 480~%# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)~%# dim[1].label  = \"width\"~%# dim[1].size   = 640~%# dim[1].stride = 3*640 = 1920~%# dim[2].label  = \"channel\"~%# dim[2].size   = 3~%# dim[2].stride = 3~%#~%# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.~%~%================================================================================~%MSG: std_msgs/MultiArrayDimension~%string label   # label of given dimension~%uint32 size    # size of given dimension (in type units)~%uint32 stride  # stride of given dimension~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <PolicyProb-request>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'obs) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
     4 (cl:length (cl:slot-value msg 'task))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <PolicyProb-request>))
  "Converts a ROS message object to a list"
  (cl:list 'PolicyProb-request
    (cl:cons ':obs (obs msg))
    (cl:cons ':task (task msg))
))
;//! \htmlinclude PolicyProb-response.msg.html

(cl:defclass <PolicyProb-response> (roslisp-msg-protocol:ros-message)
  ((mu
    :reader mu
    :initarg :mu
    :type (cl:vector std_msgs-msg:Float32MultiArray)
   :initform (cl:make-array 0 :element-type 'std_msgs-msg:Float32MultiArray :initial-element (cl:make-instance 'std_msgs-msg:Float32MultiArray)))
   (sigma
    :reader sigma
    :initarg :sigma
    :type (cl:vector std_msgs-msg:Float32MultiArray)
   :initform (cl:make-array 0 :element-type 'std_msgs-msg:Float32MultiArray :initial-element (cl:make-instance 'std_msgs-msg:Float32MultiArray))))
)

(cl:defclass PolicyProb-response (<PolicyProb-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <PolicyProb-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'PolicyProb-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tamp_ros-srv:<PolicyProb-response> is deprecated: use tamp_ros-srv:PolicyProb-response instead.")))

(cl:ensure-generic-function 'mu-val :lambda-list '(m))
(cl:defmethod mu-val ((m <PolicyProb-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tamp_ros-srv:mu-val is deprecated.  Use tamp_ros-srv:mu instead.")
  (mu m))

(cl:ensure-generic-function 'sigma-val :lambda-list '(m))
(cl:defmethod sigma-val ((m <PolicyProb-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tamp_ros-srv:sigma-val is deprecated.  Use tamp_ros-srv:sigma instead.")
  (sigma m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <PolicyProb-response>) ostream)
  "Serializes a message object of type '<PolicyProb-response>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'mu))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'mu))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'sigma))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'sigma))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <PolicyProb-response>) istream)
  "Deserializes a message object of type '<PolicyProb-response>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'mu) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'mu)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'std_msgs-msg:Float32MultiArray))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'sigma) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'sigma)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'std_msgs-msg:Float32MultiArray))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<PolicyProb-response>)))
  "Returns string type for a service object of type '<PolicyProb-response>"
  "tamp_ros/PolicyProbResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'PolicyProb-response)))
  "Returns string type for a service object of type 'PolicyProb-response"
  "tamp_ros/PolicyProbResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<PolicyProb-response>)))
  "Returns md5sum for a message object of type '<PolicyProb-response>"
  "543016ad28d3afef79460f66829d896a")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'PolicyProb-response)))
  "Returns md5sum for a message object of type 'PolicyProb-response"
  "543016ad28d3afef79460f66829d896a")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<PolicyProb-response>)))
  "Returns full string definition for message of type '<PolicyProb-response>"
  (cl:format cl:nil "~%std_msgs/Float32MultiArray[] mu~%std_msgs/Float32MultiArray[] sigma~%~%~%================================================================================~%MSG: std_msgs/Float32MultiArray~%# Please look at the MultiArrayLayout message definition for~%# documentation on all multiarrays.~%~%MultiArrayLayout  layout        # specification of data layout~%float32[]         data          # array of data~%~%~%================================================================================~%MSG: std_msgs/MultiArrayLayout~%# The multiarray declares a generic multi-dimensional array of a~%# particular data type.  Dimensions are ordered from outer most~%# to inner most.~%~%MultiArrayDimension[] dim # Array of dimension properties~%uint32 data_offset        # padding elements at front of data~%~%# Accessors should ALWAYS be written in terms of dimension stride~%# and specified outer-most dimension first.~%# ~%# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]~%#~%# A standard, 3-channel 640x480 image with interleaved color channels~%# would be specified as:~%#~%# dim[0].label  = \"height\"~%# dim[0].size   = 480~%# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)~%# dim[1].label  = \"width\"~%# dim[1].size   = 640~%# dim[1].stride = 3*640 = 1920~%# dim[2].label  = \"channel\"~%# dim[2].size   = 3~%# dim[2].stride = 3~%#~%# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.~%~%================================================================================~%MSG: std_msgs/MultiArrayDimension~%string label   # label of given dimension~%uint32 size    # size of given dimension (in type units)~%uint32 stride  # stride of given dimension~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'PolicyProb-response)))
  "Returns full string definition for message of type 'PolicyProb-response"
  (cl:format cl:nil "~%std_msgs/Float32MultiArray[] mu~%std_msgs/Float32MultiArray[] sigma~%~%~%================================================================================~%MSG: std_msgs/Float32MultiArray~%# Please look at the MultiArrayLayout message definition for~%# documentation on all multiarrays.~%~%MultiArrayLayout  layout        # specification of data layout~%float32[]         data          # array of data~%~%~%================================================================================~%MSG: std_msgs/MultiArrayLayout~%# The multiarray declares a generic multi-dimensional array of a~%# particular data type.  Dimensions are ordered from outer most~%# to inner most.~%~%MultiArrayDimension[] dim # Array of dimension properties~%uint32 data_offset        # padding elements at front of data~%~%# Accessors should ALWAYS be written in terms of dimension stride~%# and specified outer-most dimension first.~%# ~%# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]~%#~%# A standard, 3-channel 640x480 image with interleaved color channels~%# would be specified as:~%#~%# dim[0].label  = \"height\"~%# dim[0].size   = 480~%# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)~%# dim[1].label  = \"width\"~%# dim[1].size   = 640~%# dim[1].stride = 3*640 = 1920~%# dim[2].label  = \"channel\"~%# dim[2].size   = 3~%# dim[2].stride = 3~%#~%# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.~%~%================================================================================~%MSG: std_msgs/MultiArrayDimension~%string label   # label of given dimension~%uint32 size    # size of given dimension (in type units)~%uint32 stride  # stride of given dimension~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <PolicyProb-response>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'mu) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'sigma) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <PolicyProb-response>))
  "Converts a ROS message object to a list"
  (cl:list 'PolicyProb-response
    (cl:cons ':mu (mu msg))
    (cl:cons ':sigma (sigma msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'PolicyProb)))
  'PolicyProb-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'PolicyProb)))
  'PolicyProb-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'PolicyProb)))
  "Returns string type for a service object of type '<PolicyProb>"
  "tamp_ros/PolicyProb")