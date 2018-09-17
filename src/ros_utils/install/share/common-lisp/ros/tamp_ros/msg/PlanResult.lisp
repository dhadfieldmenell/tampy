; Auto-generated. Do not edit!


(cl:in-package tamp_ros-msg)


;//! \htmlinclude PlanResult.msg.html

(cl:defclass <PlanResult> (roslisp-msg-protocol:ros-message)
  ((prob_id
    :reader prob_id
    :initarg :prob_id
    :type cl:integer
    :initform 0)
   (trajectory
    :reader trajectory
    :initarg :trajectory
    :type (cl:vector tamp_ros-msg:FloatArray)
   :initform (cl:make-array 0 :element-type 'tamp_ros-msg:FloatArray :initial-element (cl:make-instance 'tamp_ros-msg:FloatArray)))
   (success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil)
   (failed_preds
    :reader failed_preds
    :initarg :failed_preds
    :type (cl:vector cl:string)
   :initform (cl:make-array 0 :element-type 'cl:string :initial-element "")))
)

(cl:defclass PlanResult (<PlanResult>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <PlanResult>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'PlanResult)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tamp_ros-msg:<PlanResult> is deprecated: use tamp_ros-msg:PlanResult instead.")))

(cl:ensure-generic-function 'prob_id-val :lambda-list '(m))
(cl:defmethod prob_id-val ((m <PlanResult>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tamp_ros-msg:prob_id-val is deprecated.  Use tamp_ros-msg:prob_id instead.")
  (prob_id m))

(cl:ensure-generic-function 'trajectory-val :lambda-list '(m))
(cl:defmethod trajectory-val ((m <PlanResult>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tamp_ros-msg:trajectory-val is deprecated.  Use tamp_ros-msg:trajectory instead.")
  (trajectory m))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <PlanResult>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tamp_ros-msg:success-val is deprecated.  Use tamp_ros-msg:success instead.")
  (success m))

(cl:ensure-generic-function 'failed_preds-val :lambda-list '(m))
(cl:defmethod failed_preds-val ((m <PlanResult>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tamp_ros-msg:failed_preds-val is deprecated.  Use tamp_ros-msg:failed_preds instead.")
  (failed_preds m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <PlanResult>) ostream)
  "Serializes a message object of type '<PlanResult>"
  (cl:let* ((signed (cl:slot-value msg 'prob_id)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 18446744073709551616) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) unsigned) ostream)
    )
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'trajectory))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'trajectory))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'failed_preds))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((__ros_str_len (cl:length ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) ele))
   (cl:slot-value msg 'failed_preds))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <PlanResult>) istream)
  "Deserializes a message object of type '<PlanResult>"
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'prob_id) (cl:if (cl:< unsigned 9223372036854775808) unsigned (cl:- unsigned 18446744073709551616))))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'trajectory) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'trajectory)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'tamp_ros-msg:FloatArray))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'failed_preds) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'failed_preds)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:aref vals i) __ros_str_idx) (cl:code-char (cl:read-byte istream))))))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<PlanResult>)))
  "Returns string type for a message object of type '<PlanResult>"
  "tamp_ros/PlanResult")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'PlanResult)))
  "Returns string type for a message object of type 'PlanResult"
  "tamp_ros/PlanResult")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<PlanResult>)))
  "Returns md5sum for a message object of type '<PlanResult>"
  "d4f5f1c50852a30db764ffda62f46133")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'PlanResult)))
  "Returns md5sum for a message object of type 'PlanResult"
  "d4f5f1c50852a30db764ffda62f46133")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<PlanResult>)))
  "Returns full string definition for message of type '<PlanResult>"
  (cl:format cl:nil "int64 prob_id~%FloatArray[] trajectory~%bool success~%string[] failed_preds~%~%~%================================================================================~%MSG: tamp_ros/FloatArray~%float32[] data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'PlanResult)))
  "Returns full string definition for message of type 'PlanResult"
  (cl:format cl:nil "int64 prob_id~%FloatArray[] trajectory~%bool success~%string[] failed_preds~%~%~%================================================================================~%MSG: tamp_ros/FloatArray~%float32[] data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <PlanResult>))
  (cl:+ 0
     8
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'trajectory) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
     1
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'failed_preds) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4 (cl:length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <PlanResult>))
  "Converts a ROS message object to a list"
  (cl:list 'PlanResult
    (cl:cons ':prob_id (prob_id msg))
    (cl:cons ':trajectory (trajectory msg))
    (cl:cons ':success (success msg))
    (cl:cons ':failed_preds (failed_preds msg))
))
