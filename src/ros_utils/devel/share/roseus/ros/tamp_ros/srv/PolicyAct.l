;; Auto-generated. Do not edit!


(when (boundp 'tamp_ros::PolicyAct)
  (if (not (find-package "TAMP_ROS"))
    (make-package "TAMP_ROS"))
  (shadow 'PolicyAct (find-package "TAMP_ROS")))
(unless (find-package "TAMP_ROS::POLICYACT")
  (make-package "TAMP_ROS::POLICYACT"))
(unless (find-package "TAMP_ROS::POLICYACTREQUEST")
  (make-package "TAMP_ROS::POLICYACTREQUEST"))
(unless (find-package "TAMP_ROS::POLICYACTRESPONSE")
  (make-package "TAMP_ROS::POLICYACTRESPONSE"))

(in-package "ROS")





(defclass tamp_ros::PolicyActRequest
  :super ros::object
  :slots (_obs _noise _task ))

(defmethod tamp_ros::PolicyActRequest
  (:init
   (&key
    ((:obs __obs) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:noise __noise) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:task __task) "")
    )
   (send-super :init)
   (setq _obs __obs)
   (setq _noise __noise)
   (setq _task (string __task))
   self)
  (:obs
   (&optional __obs)
   (if __obs (setq _obs __obs)) _obs)
  (:noise
   (&optional __noise)
   (if __noise (setq _noise __noise)) _noise)
  (:task
   (&optional __task)
   (if __task (setq _task __task)) _task)
  (:serialization-length
   ()
   (+
    ;; float32[] _obs
    (* 4    (length _obs)) 4
    ;; float32[] _noise
    (* 4    (length _noise)) 4
    ;; string _task
    4 (length _task)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32[] _obs
     (write-long (length _obs) s)
     (dotimes (i (length _obs))
       (sys::poke (elt _obs i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32[] _noise
     (write-long (length _noise) s)
     (dotimes (i (length _noise))
       (sys::poke (elt _noise i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; string _task
       (write-long (length _task) s) (princ _task s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32[] _obs
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _obs (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _obs i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;; float32[] _noise
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _noise (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _noise i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;; string _task
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _task (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass tamp_ros::PolicyActResponse
  :super ros::object
  :slots (_act ))

(defmethod tamp_ros::PolicyActResponse
  (:init
   (&key
    ((:act __act) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _act __act)
   self)
  (:act
   (&optional __act)
   (if __act (setq _act __act)) _act)
  (:serialization-length
   ()
   (+
    ;; float32[] _act
    (* 4    (length _act)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32[] _act
     (write-long (length _act) s)
     (dotimes (i (length _act))
       (sys::poke (elt _act i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32[] _act
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _act (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _act i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;;
   self)
  )

(defclass tamp_ros::PolicyAct
  :super ros::object
  :slots ())

(setf (get tamp_ros::PolicyAct :md5sum-) "e3eb5859ffc1c0de9d569f656c4594dc")
(setf (get tamp_ros::PolicyAct :datatype-) "tamp_ros/PolicyAct")
(setf (get tamp_ros::PolicyAct :request) tamp_ros::PolicyActRequest)
(setf (get tamp_ros::PolicyAct :response) tamp_ros::PolicyActResponse)

(defmethod tamp_ros::PolicyActRequest
  (:response () (instance tamp_ros::PolicyActResponse :init)))

(setf (get tamp_ros::PolicyActRequest :md5sum-) "e3eb5859ffc1c0de9d569f656c4594dc")
(setf (get tamp_ros::PolicyActRequest :datatype-) "tamp_ros/PolicyActRequest")
(setf (get tamp_ros::PolicyActRequest :definition-)
      "float32[] obs
float32[] noise
string task

---

float32[] act

")

(setf (get tamp_ros::PolicyActResponse :md5sum-) "e3eb5859ffc1c0de9d569f656c4594dc")
(setf (get tamp_ros::PolicyActResponse :datatype-) "tamp_ros/PolicyActResponse")
(setf (get tamp_ros::PolicyActResponse :definition-)
      "float32[] obs
float32[] noise
string task

---

float32[] act

")



(provide :tamp_ros/PolicyAct "e3eb5859ffc1c0de9d569f656c4594dc")


