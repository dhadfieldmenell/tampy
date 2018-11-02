; Auto-generated. Do not edit!


(cl:in-package tamp_ros-msg)


;//! \htmlinclude HLProblem.msg.html

(cl:defclass <HLProblem> (roslisp-msg-protocol:ros-message)
  ((solver_id
    :reader solver_id
    :initarg :solver_id
    :type cl:integer
    :initform 0)
   (server_id
    :reader server_id
    :initarg :server_id
    :type cl:integer
    :initform 0)
   (init_state
    :reader init_state
    :initarg :init_state
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0))
   (cond
    :reader cond
    :initarg :cond
    :type cl:integer
    :initform 0)
   (path_to
    :reader path_to
    :initarg :path_to
    :type cl:string
    :initform ""))
)

(cl:defclass HLProblem (<HLProblem>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <HLProblem>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'HLProblem)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tamp_ros-msg:<HLProblem> is deprecated: use tamp_ros-msg:HLProblem instead.")))

(cl:ensure-generic-function 'solver_id-val :lambda-list '(m))
(cl:defmethod solver_id-val ((m <HLProblem>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tamp_ros-msg:solver_id-val is deprecated.  Use tamp_ros-msg:solver_id instead.")
  (solver_id m))

(cl:ensure-generic-function 'server_id-val :lambda-list '(m))
(cl:defmethod server_id-val ((m <HLProblem>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tamp_ros-msg:server_id-val is deprecated.  Use tamp_ros-msg:server_id instead.")
  (server_id m))

(cl:ensure-generic-function 'init_state-val :lambda-list '(m))
(cl:defmethod init_state-val ((m <HLProblem>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tamp_ros-msg:init_state-val is deprecated.  Use tamp_ros-msg:init_state instead.")
  (init_state m))

(cl:ensure-generic-function 'cond-val :lambda-list '(m))
(cl:defmethod cond-val ((m <HLProblem>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tamp_ros-msg:cond-val is deprecated.  Use tamp_ros-msg:cond instead.")
  (cond m))

(cl:ensure-generic-function 'path_to-val :lambda-list '(m))
(cl:defmethod path_to-val ((m <HLProblem>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tamp_ros-msg:path_to-val is deprecated.  Use tamp_ros-msg:path_to instead.")
  (path_to m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <HLProblem>) ostream)
  "Serializes a message object of type '<HLProblem>"
  (cl:let* ((signed (cl:slot-value msg 'solver_id)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'server_id)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'init_state))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'init_state))
  (cl:let* ((signed (cl:slot-value msg 'cond)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'path_to))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'path_to))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <HLProblem>) istream)
  "Deserializes a message object of type '<HLProblem>"
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'solver_id) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'server_id) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'init_state) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'init_state)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits))))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'cond) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'path_to) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'path_to) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<HLProblem>)))
  "Returns string type for a message object of type '<HLProblem>"
  "tamp_ros/HLProblem")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'HLProblem)))
  "Returns string type for a message object of type 'HLProblem"
  "tamp_ros/HLProblem")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<HLProblem>)))
  "Returns md5sum for a message object of type '<HLProblem>"
  "1a2adc264904a8c2fffd546791b7a9ec")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'HLProblem)))
  "Returns md5sum for a message object of type 'HLProblem"
  "1a2adc264904a8c2fffd546791b7a9ec")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<HLProblem>)))
  "Returns full string definition for message of type '<HLProblem>"
  (cl:format cl:nil "int32 solver_id~%int32 server_id~%float32[] init_state~%int32 cond~%string path_to~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'HLProblem)))
  "Returns full string definition for message of type 'HLProblem"
  (cl:format cl:nil "int32 solver_id~%int32 server_id~%float32[] init_state~%int32 cond~%string path_to~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <HLProblem>))
  (cl:+ 0
     4
     4
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'init_state) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     4
     4 (cl:length (cl:slot-value msg 'path_to))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <HLProblem>))
  "Converts a ROS message object to a list"
  (cl:list 'HLProblem
    (cl:cons ':solver_id (solver_id msg))
    (cl:cons ':server_id (server_id msg))
    (cl:cons ':init_state (init_state msg))
    (cl:cons ':cond (cond msg))
    (cl:cons ':path_to (path_to msg))
))
