; Auto-generated. Do not edit!


(cl:in-package tamp_ros-srv)


;//! \htmlinclude PolicyAct-request.msg.html

(cl:defclass <PolicyAct-request> (roslisp-msg-protocol:ros-message)
  ((obs
    :reader obs
    :initarg :obs
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0))
   (noise
    :reader noise
    :initarg :noise
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0))
   (task
    :reader task
    :initarg :task
    :type cl:string
    :initform ""))
)

(cl:defclass PolicyAct-request (<PolicyAct-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <PolicyAct-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'PolicyAct-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tamp_ros-srv:<PolicyAct-request> is deprecated: use tamp_ros-srv:PolicyAct-request instead.")))

(cl:ensure-generic-function 'obs-val :lambda-list '(m))
(cl:defmethod obs-val ((m <PolicyAct-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tamp_ros-srv:obs-val is deprecated.  Use tamp_ros-srv:obs instead.")
  (obs m))

(cl:ensure-generic-function 'noise-val :lambda-list '(m))
(cl:defmethod noise-val ((m <PolicyAct-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tamp_ros-srv:noise-val is deprecated.  Use tamp_ros-srv:noise instead.")
  (noise m))

(cl:ensure-generic-function 'task-val :lambda-list '(m))
(cl:defmethod task-val ((m <PolicyAct-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tamp_ros-srv:task-val is deprecated.  Use tamp_ros-srv:task instead.")
  (task m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <PolicyAct-request>) ostream)
  "Serializes a message object of type '<PolicyAct-request>"
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
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'noise))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'noise))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'task))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'task))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <PolicyAct-request>) istream)
  "Deserializes a message object of type '<PolicyAct-request>"
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
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'noise) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'noise)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits))))))
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<PolicyAct-request>)))
  "Returns string type for a service object of type '<PolicyAct-request>"
  "tamp_ros/PolicyActRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'PolicyAct-request)))
  "Returns string type for a service object of type 'PolicyAct-request"
  "tamp_ros/PolicyActRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<PolicyAct-request>)))
  "Returns md5sum for a message object of type '<PolicyAct-request>"
  "e3eb5859ffc1c0de9d569f656c4594dc")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'PolicyAct-request)))
  "Returns md5sum for a message object of type 'PolicyAct-request"
  "e3eb5859ffc1c0de9d569f656c4594dc")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<PolicyAct-request>)))
  "Returns full string definition for message of type '<PolicyAct-request>"
  (cl:format cl:nil "float32[] obs~%float32[] noise~%string task~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'PolicyAct-request)))
  "Returns full string definition for message of type 'PolicyAct-request"
  (cl:format cl:nil "float32[] obs~%float32[] noise~%string task~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <PolicyAct-request>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'obs) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'noise) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     4 (cl:length (cl:slot-value msg 'task))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <PolicyAct-request>))
  "Converts a ROS message object to a list"
  (cl:list 'PolicyAct-request
    (cl:cons ':obs (obs msg))
    (cl:cons ':noise (noise msg))
    (cl:cons ':task (task msg))
))
;//! \htmlinclude PolicyAct-response.msg.html

(cl:defclass <PolicyAct-response> (roslisp-msg-protocol:ros-message)
  ((act
    :reader act
    :initarg :act
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0)))
)

(cl:defclass PolicyAct-response (<PolicyAct-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <PolicyAct-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'PolicyAct-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tamp_ros-srv:<PolicyAct-response> is deprecated: use tamp_ros-srv:PolicyAct-response instead.")))

(cl:ensure-generic-function 'act-val :lambda-list '(m))
(cl:defmethod act-val ((m <PolicyAct-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tamp_ros-srv:act-val is deprecated.  Use tamp_ros-srv:act instead.")
  (act m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <PolicyAct-response>) ostream)
  "Serializes a message object of type '<PolicyAct-response>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'act))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'act))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <PolicyAct-response>) istream)
  "Deserializes a message object of type '<PolicyAct-response>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'act) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'act)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits))))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<PolicyAct-response>)))
  "Returns string type for a service object of type '<PolicyAct-response>"
  "tamp_ros/PolicyActResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'PolicyAct-response)))
  "Returns string type for a service object of type 'PolicyAct-response"
  "tamp_ros/PolicyActResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<PolicyAct-response>)))
  "Returns md5sum for a message object of type '<PolicyAct-response>"
  "e3eb5859ffc1c0de9d569f656c4594dc")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'PolicyAct-response)))
  "Returns md5sum for a message object of type 'PolicyAct-response"
  "e3eb5859ffc1c0de9d569f656c4594dc")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<PolicyAct-response>)))
  "Returns full string definition for message of type '<PolicyAct-response>"
  (cl:format cl:nil "~%float32[] act~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'PolicyAct-response)))
  "Returns full string definition for message of type 'PolicyAct-response"
  (cl:format cl:nil "~%float32[] act~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <PolicyAct-response>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'act) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <PolicyAct-response>))
  "Converts a ROS message object to a list"
  (cl:list 'PolicyAct-response
    (cl:cons ':act (act msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'PolicyAct)))
  'PolicyAct-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'PolicyAct)))
  'PolicyAct-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'PolicyAct)))
  "Returns string type for a service object of type '<PolicyAct>"
  "tamp_ros/PolicyAct")