;; Auto-generated. Do not edit!


(when (boundp 'tamp_ros::Primitive)
  (if (not (find-package "TAMP_ROS"))
    (make-package "TAMP_ROS"))
  (shadow 'Primitive (find-package "TAMP_ROS")))
(unless (find-package "TAMP_ROS::PRIMITIVE")
  (make-package "TAMP_ROS::PRIMITIVE"))
(unless (find-package "TAMP_ROS::PRIMITIVEREQUEST")
  (make-package "TAMP_ROS::PRIMITIVEREQUEST"))
(unless (find-package "TAMP_ROS::PRIMITIVERESPONSE")
  (make-package "TAMP_ROS::PRIMITIVERESPONSE"))

(in-package "ROS")





(defclass tamp_ros::PrimitiveRequest
  :super ros::object
  :slots (_prim_obs ))

(defmethod tamp_ros::PrimitiveRequest
  (:init
   (&key
    ((:prim_obs __prim_obs) 0.0)
    )
   (send-super :init)
   (setq _prim_obs (float __prim_obs))
   self)
  (:prim_obs
   (&optional __prim_obs)
   (if __prim_obs (setq _prim_obs __prim_obs)) _prim_obs)
  (:serialization-length
   ()
   (+
    ;; float32 _prim_obs
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _prim_obs
       (sys::poke _prim_obs (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _prim_obs
     (setq _prim_obs (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(defclass tamp_ros::PrimitiveResponse
  :super ros::object
  :slots (_task_distr _obj_distr _targ_distr ))

(defmethod tamp_ros::PrimitiveResponse
  (:init
   (&key
    ((:task_distr __task_distr) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:obj_distr __obj_distr) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:targ_distr __targ_distr) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _task_distr __task_distr)
   (setq _obj_distr __obj_distr)
   (setq _targ_distr __targ_distr)
   self)
  (:task_distr
   (&optional __task_distr)
   (if __task_distr (setq _task_distr __task_distr)) _task_distr)
  (:obj_distr
   (&optional __obj_distr)
   (if __obj_distr (setq _obj_distr __obj_distr)) _obj_distr)
  (:targ_distr
   (&optional __targ_distr)
   (if __targ_distr (setq _targ_distr __targ_distr)) _targ_distr)
  (:serialization-length
   ()
   (+
    ;; float32[] _task_distr
    (* 4    (length _task_distr)) 4
    ;; float32[] _obj_distr
    (* 4    (length _obj_distr)) 4
    ;; float32[] _targ_distr
    (* 4    (length _targ_distr)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32[] _task_distr
     (write-long (length _task_distr) s)
     (dotimes (i (length _task_distr))
       (sys::poke (elt _task_distr i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32[] _obj_distr
     (write-long (length _obj_distr) s)
     (dotimes (i (length _obj_distr))
       (sys::poke (elt _obj_distr i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32[] _targ_distr
     (write-long (length _targ_distr) s)
     (dotimes (i (length _targ_distr))
       (sys::poke (elt _targ_distr i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32[] _task_distr
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _task_distr (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _task_distr i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;; float32[] _obj_distr
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _obj_distr (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _obj_distr i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;; float32[] _targ_distr
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _targ_distr (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _targ_distr i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;;
   self)
  )

(defclass tamp_ros::Primitive
  :super ros::object
  :slots ())

(setf (get tamp_ros::Primitive :md5sum-) "ec8948c09b640bcf5ec37fe64f2d51b1")
(setf (get tamp_ros::Primitive :datatype-) "tamp_ros/Primitive")
(setf (get tamp_ros::Primitive :request) tamp_ros::PrimitiveRequest)
(setf (get tamp_ros::Primitive :response) tamp_ros::PrimitiveResponse)

(defmethod tamp_ros::PrimitiveRequest
  (:response () (instance tamp_ros::PrimitiveResponse :init)))

(setf (get tamp_ros::PrimitiveRequest :md5sum-) "ec8948c09b640bcf5ec37fe64f2d51b1")
(setf (get tamp_ros::PrimitiveRequest :datatype-) "tamp_ros/PrimitiveRequest")
(setf (get tamp_ros::PrimitiveRequest :definition-)
      "float32 prim_obs

---

float32[] task_distr
float32[] obj_distr
float32[] targ_distr

")

(setf (get tamp_ros::PrimitiveResponse :md5sum-) "ec8948c09b640bcf5ec37fe64f2d51b1")
(setf (get tamp_ros::PrimitiveResponse :datatype-) "tamp_ros/PrimitiveResponse")
(setf (get tamp_ros::PrimitiveResponse :definition-)
      "float32 prim_obs

---

float32[] task_distr
float32[] obj_distr
float32[] targ_distr

")



(provide :tamp_ros/Primitive "ec8948c09b640bcf5ec37fe64f2d51b1")


