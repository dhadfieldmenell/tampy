; Auto-generated. Do not edit!


(cl:in-package tamp_ros-srv)


;//! \htmlinclude Primitive-request.msg.html

(cl:defclass <Primitive-request> (roslisp-msg-protocol:ros-message)
  ((prim_obs
    :reader prim_obs
    :initarg :prim_obs
    :type cl:float
    :initform 0.0))
)

(cl:defclass Primitive-request (<Primitive-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Primitive-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Primitive-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tamp_ros-srv:<Primitive-request> is deprecated: use tamp_ros-srv:Primitive-request instead.")))

(cl:ensure-generic-function 'prim_obs-val :lambda-list '(m))
(cl:defmethod prim_obs-val ((m <Primitive-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tamp_ros-srv:prim_obs-val is deprecated.  Use tamp_ros-srv:prim_obs instead.")
  (prim_obs m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Primitive-request>) ostream)
  "Serializes a message object of type '<Primitive-request>"
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'prim_obs))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Primitive-request>) istream)
  "Deserializes a message object of type '<Primitive-request>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'prim_obs) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Primitive-request>)))
  "Returns string type for a service object of type '<Primitive-request>"
  "tamp_ros/PrimitiveRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Primitive-request)))
  "Returns string type for a service object of type 'Primitive-request"
  "tamp_ros/PrimitiveRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Primitive-request>)))
  "Returns md5sum for a message object of type '<Primitive-request>"
  "ec8948c09b640bcf5ec37fe64f2d51b1")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Primitive-request)))
  "Returns md5sum for a message object of type 'Primitive-request"
  "ec8948c09b640bcf5ec37fe64f2d51b1")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Primitive-request>)))
  "Returns full string definition for message of type '<Primitive-request>"
  (cl:format cl:nil "float32 prim_obs~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Primitive-request)))
  "Returns full string definition for message of type 'Primitive-request"
  (cl:format cl:nil "float32 prim_obs~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Primitive-request>))
  (cl:+ 0
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Primitive-request>))
  "Converts a ROS message object to a list"
  (cl:list 'Primitive-request
    (cl:cons ':prim_obs (prim_obs msg))
))
;//! \htmlinclude Primitive-response.msg.html

(cl:defclass <Primitive-response> (roslisp-msg-protocol:ros-message)
  ((task_distr
    :reader task_distr
    :initarg :task_distr
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0))
   (obj_distr
    :reader obj_distr
    :initarg :obj_distr
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0))
   (targ_distr
    :reader targ_distr
    :initarg :targ_distr
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0)))
)

(cl:defclass Primitive-response (<Primitive-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Primitive-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Primitive-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tamp_ros-srv:<Primitive-response> is deprecated: use tamp_ros-srv:Primitive-response instead.")))

(cl:ensure-generic-function 'task_distr-val :lambda-list '(m))
(cl:defmethod task_distr-val ((m <Primitive-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tamp_ros-srv:task_distr-val is deprecated.  Use tamp_ros-srv:task_distr instead.")
  (task_distr m))

(cl:ensure-generic-function 'obj_distr-val :lambda-list '(m))
(cl:defmethod obj_distr-val ((m <Primitive-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tamp_ros-srv:obj_distr-val is deprecated.  Use tamp_ros-srv:obj_distr instead.")
  (obj_distr m))

(cl:ensure-generic-function 'targ_distr-val :lambda-list '(m))
(cl:defmethod targ_distr-val ((m <Primitive-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tamp_ros-srv:targ_distr-val is deprecated.  Use tamp_ros-srv:targ_distr instead.")
  (targ_distr m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Primitive-response>) ostream)
  "Serializes a message object of type '<Primitive-response>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'task_distr))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'task_distr))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'obj_distr))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'obj_distr))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'targ_distr))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'targ_distr))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Primitive-response>) istream)
  "Deserializes a message object of type '<Primitive-response>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'task_distr) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'task_distr)))
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
  (cl:setf (cl:slot-value msg 'obj_distr) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'obj_distr)))
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
  (cl:setf (cl:slot-value msg 'targ_distr) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'targ_distr)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits))))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Primitive-response>)))
  "Returns string type for a service object of type '<Primitive-response>"
  "tamp_ros/PrimitiveResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Primitive-response)))
  "Returns string type for a service object of type 'Primitive-response"
  "tamp_ros/PrimitiveResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Primitive-response>)))
  "Returns md5sum for a message object of type '<Primitive-response>"
  "ec8948c09b640bcf5ec37fe64f2d51b1")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Primitive-response)))
  "Returns md5sum for a message object of type 'Primitive-response"
  "ec8948c09b640bcf5ec37fe64f2d51b1")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Primitive-response>)))
  "Returns full string definition for message of type '<Primitive-response>"
  (cl:format cl:nil "~%float32[] task_distr~%float32[] obj_distr~%float32[] targ_distr~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Primitive-response)))
  "Returns full string definition for message of type 'Primitive-response"
  (cl:format cl:nil "~%float32[] task_distr~%float32[] obj_distr~%float32[] targ_distr~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Primitive-response>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'task_distr) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'obj_distr) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'targ_distr) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Primitive-response>))
  "Converts a ROS message object to a list"
  (cl:list 'Primitive-response
    (cl:cons ':task_distr (task_distr msg))
    (cl:cons ':obj_distr (obj_distr msg))
    (cl:cons ':targ_distr (targ_distr msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'Primitive)))
  'Primitive-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'Primitive)))
  'Primitive-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Primitive)))
  "Returns string type for a service object of type '<Primitive>"
  "tamp_ros/Primitive")