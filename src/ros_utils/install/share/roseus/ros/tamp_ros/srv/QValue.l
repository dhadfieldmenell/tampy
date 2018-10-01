;; Auto-generated. Do not edit!


(when (boundp 'tamp_ros::QValue)
  (if (not (find-package "TAMP_ROS"))
    (make-package "TAMP_ROS"))
  (shadow 'QValue (find-package "TAMP_ROS")))
(unless (find-package "TAMP_ROS::QVALUE")
  (make-package "TAMP_ROS::QVALUE"))
(unless (find-package "TAMP_ROS::QVALUEREQUEST")
  (make-package "TAMP_ROS::QVALUEREQUEST"))
(unless (find-package "TAMP_ROS::QVALUERESPONSE")
  (make-package "TAMP_ROS::QVALUERESPONSE"))

(in-package "ROS")





(defclass tamp_ros::QValueRequest
  :super ros::object
  :slots (_obs ))

(defmethod tamp_ros::QValueRequest
  (:init
   (&key
    ((:obs __obs) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _obs __obs)
   self)
  (:obs
   (&optional __obs)
   (if __obs (setq _obs __obs)) _obs)
  (:serialization-length
   ()
   (+
    ;; float32[] _obs
    (* 4    (length _obs)) 4
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
   ;;
   self)
  )

(defclass tamp_ros::QValueResponse
  :super ros::object
  :slots (_value ))

(defmethod tamp_ros::QValueResponse
  (:init
   (&key
    ((:value __value) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _value __value)
   self)
  (:value
   (&optional __value)
   (if __value (setq _value __value)) _value)
  (:serialization-length
   ()
   (+
    ;; float32[] _value
    (* 4    (length _value)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32[] _value
     (write-long (length _value) s)
     (dotimes (i (length _value))
       (sys::poke (elt _value i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32[] _value
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _value (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _value i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;;
   self)
  )

(defclass tamp_ros::QValue
  :super ros::object
  :slots ())

(setf (get tamp_ros::QValue :md5sum-) "69d0eb61126056c55900069d173c5835")
(setf (get tamp_ros::QValue :datatype-) "tamp_ros/QValue")
(setf (get tamp_ros::QValue :request) tamp_ros::QValueRequest)
(setf (get tamp_ros::QValue :response) tamp_ros::QValueResponse)

(defmethod tamp_ros::QValueRequest
  (:response () (instance tamp_ros::QValueResponse :init)))

(setf (get tamp_ros::QValueRequest :md5sum-) "69d0eb61126056c55900069d173c5835")
(setf (get tamp_ros::QValueRequest :datatype-) "tamp_ros/QValueRequest")
(setf (get tamp_ros::QValueRequest :definition-)
      "float32[] obs

---

float32[] value

")

(setf (get tamp_ros::QValueResponse :md5sum-) "69d0eb61126056c55900069d173c5835")
(setf (get tamp_ros::QValueResponse :datatype-) "tamp_ros/QValueResponse")
(setf (get tamp_ros::QValueResponse :definition-)
      "float32[] obs

---

float32[] value

")



(provide :tamp_ros/QValue "69d0eb61126056c55900069d173c5835")


