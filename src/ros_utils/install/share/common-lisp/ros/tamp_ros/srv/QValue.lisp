; Auto-generated. Do not edit!


(cl:in-package tamp_ros-srv)


;//! \htmlinclude QValue-request.msg.html

(cl:defclass <QValue-request> (roslisp-msg-protocol:ros-message)
  ((obs
    :reader obs
    :initarg :obs
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0)))
)

(cl:defclass QValue-request (<QValue-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <QValue-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'QValue-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tamp_ros-srv:<QValue-request> is deprecated: use tamp_ros-srv:QValue-request instead.")))

(cl:ensure-generic-function 'obs-val :lambda-list '(m))
(cl:defmethod obs-val ((m <QValue-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tamp_ros-srv:obs-val is deprecated.  Use tamp_ros-srv:obs instead.")
  (obs m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <QValue-request>) ostream)
  "Serializes a message object of type '<QValue-request>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'obs))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'obs))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <QValue-request>) istream)
  "Deserializes a message object of type '<QValue-request>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'obs) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'obs)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits))))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<QValue-request>)))
  "Returns string type for a service object of type '<QValue-request>"
  "tamp_ros/QValueRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'QValue-request)))
  "Returns string type for a service object of type 'QValue-request"
  "tamp_ros/QValueRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<QValue-request>)))
  "Returns md5sum for a message object of type '<QValue-request>"
  "69d0eb61126056c55900069d173c5835")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'QValue-request)))
  "Returns md5sum for a message object of type 'QValue-request"
  "69d0eb61126056c55900069d173c5835")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<QValue-request>)))
  "Returns full string definition for message of type '<QValue-request>"
  (cl:format cl:nil "float32[] obs~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'QValue-request)))
  "Returns full string definition for message of type 'QValue-request"
  (cl:format cl:nil "float32[] obs~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <QValue-request>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'obs) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <QValue-request>))
  "Converts a ROS message object to a list"
  (cl:list 'QValue-request
    (cl:cons ':obs (obs msg))
))
;//! \htmlinclude QValue-response.msg.html

(cl:defclass <QValue-response> (roslisp-msg-protocol:ros-message)
  ((value
    :reader value
    :initarg :value
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0)))
)

(cl:defclass QValue-response (<QValue-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <QValue-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'QValue-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tamp_ros-srv:<QValue-response> is deprecated: use tamp_ros-srv:QValue-response instead.")))

(cl:ensure-generic-function 'value-val :lambda-list '(m))
(cl:defmethod value-val ((m <QValue-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tamp_ros-srv:value-val is deprecated.  Use tamp_ros-srv:value instead.")
  (value m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <QValue-response>) ostream)
  "Serializes a message object of type '<QValue-response>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'value))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'value))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <QValue-response>) istream)
  "Deserializes a message object of type '<QValue-response>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'value) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'value)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits))))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<QValue-response>)))
  "Returns string type for a service object of type '<QValue-response>"
  "tamp_ros/QValueResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'QValue-response)))
  "Returns string type for a service object of type 'QValue-response"
  "tamp_ros/QValueResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<QValue-response>)))
  "Returns md5sum for a message object of type '<QValue-response>"
  "69d0eb61126056c55900069d173c5835")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'QValue-response)))
  "Returns md5sum for a message object of type 'QValue-response"
  "69d0eb61126056c55900069d173c5835")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<QValue-response>)))
  "Returns full string definition for message of type '<QValue-response>"
  (cl:format cl:nil "~%float32[] value~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'QValue-response)))
  "Returns full string definition for message of type 'QValue-response"
  (cl:format cl:nil "~%float32[] value~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <QValue-response>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'value) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <QValue-response>))
  "Converts a ROS message object to a list"
  (cl:list 'QValue-response
    (cl:cons ':value (value msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'QValue)))
  'QValue-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'QValue)))
  'QValue-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'QValue)))
  "Returns string type for a service object of type '<QValue>"
  "tamp_ros/QValue")